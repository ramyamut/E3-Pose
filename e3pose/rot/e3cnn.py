"""
Adapted from https://github.com/SCAN-NRAD/e3nn_Unet
"""
import math
from functools import partial
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import BatchNorm, Gate, Dropout
from e3nn.o3 import Irreps, Linear, FullyConnectedTensorProduct
from e3nn.math import soft_unit_step
from nnunet.network_architecture.neural_network import SegmentationNetwork

class ConvNet(SegmentationNetwork):
    def __init__(self, input_irreps, output_irreps, diameter, num_radial_basis, steps, batch_norm='instance', n=2, n_downsample =2, equivariance = 'SO3',
        lmax = 2, down_op = 'maxpool3d', stride = 2, scale =2,is_bias = True,scalar_upsampling=False,dropout_prob=0,cutoff=False, return_fmaps=False):
        """E3CNN Network Architecture

        Parameters
        ----------
        input_irreps : str
            input representations
            example: "1x0e" when one channel of scalar values
        output_irreps : str
            output representations
            example: "4x0e" when four channels of scalar values
        n_classes_vector : int
            number of vector classes
        diameter : float
            diameter of input convolution kernel in physical units
        num_radial_basis : int
            number of radial basis functions
        steps : float
            physical dimension of a pixel in physical units
        batch_norm : str, optional
            normalization: can be 'batch', 'instance' or 'None'.
            by default 'instance'
        n : int, optional
            multiplication factor of number of irreps
            between successive convolution blocks, by default 2
        n_downsample : int, optional
            number of downsampling operations, by default 2
        equivariance : str, optional
            type of equivariance, can be 'O3' or 'SO3'
            by default 'SO3'
        lmax : int, optional
            maximum spherical harmonics l
            by default 2
        down_op : str, optional
            type of downsampling operation
            can be 'maxpool3d', 'average' or 'lowpass'
            by default 'maxpool3d'
        stride : int, optional
            stride size, by default 2
        scale : int, optional
            size of pooling diameter
            in physical units, by default 2
        is_bias : bool, optional
            defines whether or not to add a bias, by default True
        scalar_upsampling : bool, optional
            flag to use scalar_upsampling, by default False
        dropout_prob : float, optional
            dropout probability between 0 and 1.0, by default 0
        cutoff: bool, optional
            cutoff basis functions at 0 outside of [0,r], by default False

        """
        super().__init__()

        self.n_classes_scalar = Irreps(output_irreps).count('0e')
        self.num_classes = self.n_classes_scalar
        
        self.n_downsample = n_downsample
        self.conv_op = nn.Conv3d #Needed in order to use nnUnet predict_3D
        
        self.return_fmaps = return_fmaps

        assert batch_norm in ['None','batch','instance'], "batch_norm needs to be 'batch', 'instance', or 'None'"
        assert down_op in ['maxpool3d','average','lowpass'], "down_op needs to be 'maxpool3d', 'average', or 'lowpass'"

        if down_op == 'lowpass':
            self.odd_resize = True
       
        else:
            self.odd_resize = False

        if equivariance == 'SO3':
            activation = [torch.relu]
            irreps_sh = Irreps.spherical_harmonics(lmax, -1)
            ne = n
            no = 0
        elif equivariance == 'O3':
            activation = [torch.relu,torch.tanh]
            irreps_sh = Irreps.spherical_harmonics(lmax, -1)
            ne = n
            no = n
        scales = [scale*2**i for i in range(n_downsample)] #TODO change 2 to variable factor
        diameters = [diameter*2**i for i in range(n_downsample+1)] #TODO change 2 to variable factor

        steps_array = [steps]
        for i in range(n_downsample):
            
            output_steps = []
            for step in steps:
                if step < scales[i]:
                    kernel_dim = math.floor(scales[i]/step)
                    output_steps.append(kernel_dim*step)
                else:
                    output_steps.append(step)

            steps_array.append(tuple(output_steps))
        self.down = Down(n_downsample,activation,irreps_sh,ne,no,batch_norm,input_irreps,diameters,num_radial_basis,steps_array,down_op,scales,stride,dropout_prob,cutoff)
        self.out = Convolution(self.down.down_irreps_out[-1], output_irreps, irreps_sh, diameter,num_radial_basis,steps,cutoff=cutoff)

        if len(output_irreps.split(' + '))>1:
            self.max_pool = AdaptiveDynamicPool3d('maxpool3d', Irreps(output_irreps.split(' + ')[0]))
            self.norm_pool = AdaptiveDynamicPool3d('norm_weighted', Irreps(output_irreps.split(' + ')[1]))
        else:
            self.max_pool = None
            self.norm_pool = AdaptiveDynamicPool3d('norm_weighted', Irreps(output_irreps))

        if is_bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self.n_classes_scalar))
        else:
            self.register_parameter('bias', None)
    
    def pool(self, x):
        if self.max_pool is not None:
            max_pool = self.max_pool(x[:,:3])
            norm_pool = self.norm_pool(x[:,3:])
            return torch.cat([max_pool, norm_pool], dim=1)
        else:
            return self.norm_pool(x)


    def forward(self, x, pool=True):
        if self.return_fmaps:
            return self.forward_fmaps(x)
        down_ftrs = self.down(x)
        x = down_ftrs[-1]
        x = self.out(x)
        if pool:
            out = self.pool(x)
        else:
            out = x
        if self.bias is not None:
            bias = self.bias.reshape(-1, 1, 1, 1)
            out = torch.cat([out[:, :self.n_classes_scalar,...] + bias, out[:, self.n_classes_scalar:,...]], dim=1)
        

        return out
    
    def forward_fmaps(self, x):

        down_ftrs = self.down(x)
        x = down_ftrs[-1]
        x = self.out(x)
        out = self.pool(x)

        if self.bias is not None:
            bias = self.bias.reshape(-1, 1, 1, 1)
            out = torch.cat([out[:, :self.n_classes_scalar,...] + bias, out[:, self.n_classes_scalar:,...]], dim=1)
        
        fmaps = down_ftrs + [x] + [out]
        
        return fmaps

class Down(nn.Module):
    """
    E3-CNN Encoder Architecture
    """

    def __init__(self, n_downsample,activation,irreps_sh,ne,no,BN,input,diameters,num_radial_basis,steps,down_op,scale,stride,dropout_prob,cutoff):
        super().__init__()

        blocks = []
        self.down_irreps_out = []

        for n in range(n_downsample+1):
            irreps_hidden = Irreps(f"{4*ne}x0e + {4*no}x0o + {2*ne}x1e +  {2*no}x1o + {ne}x2e + {no}x2o").simplify()
            block = ConvolutionBlock(input,irreps_hidden,activation,irreps_sh,BN, diameters[n],num_radial_basis,steps[n],dropout_prob,cutoff, transpose=False)
            blocks.append(block)
            self.down_irreps_out.append(block.irreps_out)
            input = block.irreps_out
            ne *= 2
            no *= 2

        self.down_blocks = nn.ModuleList(blocks)

        pooling = []
        for n in range(n_downsample):
            pooling.append(DynamicPool3d(scale[n],steps[n],down_op,self.down_irreps_out[n]))

        self.down_pool = nn.ModuleList(pooling)

    def forward(self, x):
        ftrs = []
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            #num_scalar_feats = self.down_irreps_out[i][0][0]
            #x = softmax_activation(x, num_scalar_feats)
            ftrs.append(x)
            if i < len(self.down_blocks)-1:
                x = self.down_pool[i](x)
        return ftrs

class ConvolutionBlock(nn.Module):
    """
    Convolution block class that uses equivariant convolutions
    """
    def __init__(self, input, irreps_hidden, activation, irreps_sh, normalization,diameter,num_radial_basis,steps,dropout_prob,cutoff,transpose):
        super().__init__()

        if normalization == 'None':
            BN = Identity
        elif normalization == 'batch':
            BN = BatchNorm
        elif normalization == 'instance':
            BN = partial(BatchNorm,instance=True)

        irreps_scalars = Irreps( [ (mul, ir) for mul, ir in irreps_hidden if ir.l == 0 ] )
        irreps_gated   = Irreps( [ (mul, ir) for mul, ir in irreps_hidden if ir.l > 0  ] )
        irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")

        #fe = sum(mul for mul,ir in irreps_gated if ir.p == 1)
        #fo = sum(mul for mul,ir in irreps_gated if ir.p == -1)
        #irreps_gates = Irreps(f"{fe}x0e+{fo}x0o").simplify()
        if irreps_gates.dim == 0:
            irreps_gates = irreps_gates.simplify()
            activation_gate = []
        else:
            activation_gate = [torch.sigmoid]
            #activation_gate = [torch.sigmoid, torch.tanh][:len(activation)]
        self.gate1 = Gate(irreps_scalars, activation, irreps_gates, activation_gate, irreps_gated)
        self.conv1 = Convolution(input, self.gate1.irreps_in, irreps_sh, diameter,num_radial_basis,steps,cutoff=cutoff, transpose=transpose)
        self.batchnorm1 = BN(self.gate1.irreps_in)
        self.dropout1 = Dropout(self.gate1.irreps_out, dropout_prob)

        self.gate2 = Gate(irreps_scalars, activation, irreps_gates, activation_gate, irreps_gated)
        self.conv2 = Convolution(self.gate1.irreps_out, self.gate2.irreps_in, irreps_sh, diameter,num_radial_basis,steps,cutoff=cutoff, transpose=transpose)
        self.batchnorm2 = BN(self.gate2.irreps_in)
        self.dropout2 = Dropout(self.gate2.irreps_out, dropout_prob)

        self.irreps_out = self.gate2.irreps_out

    def forward(self, x):
 
        x = self.conv1(x)
        x = self.batchnorm1(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate1(x.transpose(1, 4)).transpose(1, 4)
        x = self.dropout1(x.transpose(1, 4)).transpose(1, 4)

        x = self.conv2(x)
        x = self.batchnorm2(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate2(x.transpose(1, 4)).transpose(1, 4)
        x = self.dropout2(x.transpose(1, 4)).transpose(1, 4)
        return x

class Convolution(torch.nn.Module):
    r"""Implementation of equivariant convolutions

    Parameters
    ----------
    irreps_in : `Irreps`
        input irreps

    irreps_out : `Irreps`
        output irreps

    irreps_sh : `Irreps`
        set typically to ``o3.Irreps.spherical_harmonics(lmax)``

    diameter : float
        diameter of the filter in physical units

    num_radial_basis : int
        number of radial basis functions

    steps : tuple of float
        size of the pixel in physical units
    """
    def __init__(self, irreps_in, irreps_out, irreps_sh, diameter, num_radial_basis, steps=(1.0, 1.0, 1.0),cutoff=True, transpose=False, **kwargs):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)

        self.num_radial_basis = num_radial_basis
        self.transpose = transpose

        # self-connection
        self.sc = Linear(self.irreps_in, self.irreps_out) # first transform each input irrep to same space as output irrep

        # connection with neighbors
        r = diameter / 2

        s = math.floor(r / steps[0])
        x = torch.arange(-s, s + 1.0) * steps[0]

        s = math.floor(r / steps[1])
        y = torch.arange(-s, s + 1.0) * steps[1]

        s = math.floor(r / steps[2])
        z = torch.arange(-s, s + 1.0) * steps[2]

        lattice = torch.stack(torch.meshgrid(x, y, z), dim=-1)  # [x, y, z, R^3]
        self.register_buffer('lattice', lattice)

        if 'padding' not in kwargs:
            kwargs['padding'] = tuple(s // 2 for s in lattice.shape[:3])
        self.kwargs = kwargs

        emb = soft_one_hot_linspace(
            x=lattice.norm(dim=-1), # [x, y, z]
            start=0.0,
            end=r,
            number=self.num_radial_basis,
            basis='smooth_finite',
            cutoff=cutoff,
        ) # [x, y, z, B] for smooth finite, B is 5
        self.register_buffer('emb', emb)

        sh = o3.spherical_harmonics(
            l=self.irreps_sh, # 1x0e, 1x1e, 1x2e (why do we use e for l=1?s) this should be 4 i think?
            x=lattice,
            normalize=True,
            normalization='component'
        )  # [x, y, z, irreps_sh.dim]
        self.register_buffer('sh', sh) # [x,y,z,9]

        self.tp = FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False,compile_right=True)

        self.weight = torch.nn.Parameter(torch.randn(self.num_radial_basis, self.tp.weight_numel))

    def kernel(self):
        weight = self.emb @ self.weight #[s, s, s, N], N learned radial kernels
        weight = weight / (self.sh.shape[0] * self.sh.shape[1] * self.sh.shape[2]) # normalize... why?
        kernel = self.tp.right(self.sh, weight)  # [x, y, z, irreps_in.dim, irreps_out.dim]
        kernel = torch.einsum('xyzio->oixyz', kernel) # [irreps_out.dim, irreps_in.dim, x, y, z]
        return kernel

    def forward(self, x):
        r"""
        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``(batch, irreps_in.dim, x, y, z)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, irreps_out.dim, x, y, z)``
        """
        sc = self.sc(x.transpose(1, 4)).transpose(1, 4)
        
        if self.transpose:
            out = sc + torch.nn.functional.conv_transpose3d(x, self.kernel().transpose(0, 1), **self.kwargs)
        else:
            out = sc + torch.nn.functional.conv3d(x, self.kernel(), **self.kwargs)

        return out

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x):
        return x

"""
POOLING CLASSES/FUNCTIONS
"""
class DynamicPool3d(torch.nn.Module):
    def __init__(self, scale, steps, mode, irreps):
        super().__init__()

        self.scale = scale #in physical units
        self.steps = steps
        self.mode = mode
        self.kernel_size = tuple([math.floor(self.scale/step) if step < self.scale else 1 for step in self.steps])
        self.irreps = irreps

    def forward(self, input):

        if self.mode == 'maxpool3d':
        
            out = max_pool3d(input, self.irreps, self.kernel_size, stride=self.kernel_size) #e3nn max_pool3d implementation
            #out = F.max_pool3d(input, self.kernel_size, stride=self.kernel_size) #non-equivariant pytorch implementation
        elif self.mode == 'average':
            out = F.avg_pool3d(input, self.kernel_size, stride=self.kernel_size)

        return out

class AdaptiveDynamicPool3d(torch.nn.Module):
    def __init__(self, mode, irreps):
        super().__init__()

        self.mode = mode
        self.irreps = irreps

    def forward(self, input):
        kernel_size = (input.shape[2], input.shape[3], input.shape[4])
        if self.mode == 'maxpool3d':
        
            out = max_pool3d(input, self.irreps, kernel_size, stride=kernel_size) #e3nn max_pool3d implementation
            #out = F.max_pool3d(input, self.kernel_size, stride=self.kernel_size) #non-equivariant pytorch implementation
        elif self.mode == 'average':
            out = F.avg_pool3d(input, kernel_size, stride=kernel_size)
        elif self.mode == 'norm_weighted':
            out = adaptive_norm_weighted_pool3d(input, self.irreps, kernel_size, stride=kernel_size)

        return out

def max_pool3d(input, irreps, kernel_size, stride):

    assert input.shape[1] == irreps.dim, "Shape mismatch"
    cat_list = []

    start = 0
    for i in irreps.ls:

        end = start + 2*i+1
        temp = input[:,start:end,...]
        if i == 0:
            pooled,indices = F.max_pool3d_with_indices(temp[:,0,...],kernel_size,stride=stride,return_indices=True)
            cat_list.append(pooled)
        else:
            pooled, indices = F.max_pool3d_with_indices(temp.norm(dim = 1),kernel_size,stride=stride,return_indices=True)
            for slice in range(2*i+1):
                pooled = temp[:,slice,...].flatten()[indices]
                cat_list.append(pooled)
        start = end

    return torch.stack(tuple(cat_list),dim = 1)

def max_pool3d_optimized(input, irreps, kernel_size, stride):

    assert input.shape[1] == irreps.dim, "Shape mismatch"
    cat_list = []

    start = 0
    for i in irreps.ls:
        # x = [batch, mul * dim, x, y, z]
        # x = [batch, mul, dim, x, y, z]
        # norm = [batch, mul, x, y, z]
        # indices = [batch, mul, x, z, y]
        # x = x.transpose(0, 2)  [dim, batch, mul, x, y, z]
        # x[:, indices]  [dim, batch, mul, x, y, z]
        #
        # x.transpose(0, 1).flatten(1)[:, i].transpose(0, 1).shape
        # [6]: _, i = torch.nn.functional.max_pool2d_with_indices(x.pow(2).sum(1), 2, stride=2, return_indices=True)

        end = start + 2*i+1
        temp = input[:,start:end,...]
        if i == 0:
            pooled,indices = F.max_pool3d_with_indices(temp[:,0,...],kernel_size,stride=stride,return_indices=True)
            cat_list.append(pooled)
        else:
            pooled, indices = F.max_pool3d_with_indices(temp.norm(dim = 1),kernel_size,stride=stride,return_indices=True)
            for slice in range(2*i+1):
                pooled = temp[:,slice,...].flatten()[indices]
                cat_list.append(pooled)
        start = end

    return torch.stack(tuple(cat_list),dim = 1)

def max_pool3d(input, irreps, kernel_size, stride):

    assert input.shape[1] == irreps.dim, "Shape mismatch"
    cat_list = []

    start = 0
    for i in irreps.ls:

        end = start + 2*i+1
        temp = input[:,start:end,...]
        if i == 0:
            pooled,indices = F.max_pool3d_with_indices(temp[:,0,...],kernel_size,stride=stride,return_indices=True)
            cat_list.append(pooled)
        else:
            pooled, indices = F.max_pool3d_with_indices(temp.norm(dim = 1),kernel_size,stride=stride,return_indices=True)
            for slice in range(2*i+1):
                pooled = temp[:,slice,...].flatten()[indices]
                cat_list.append(pooled)
        start = end

    return torch.stack(tuple(cat_list),dim = 1)

def adaptive_norm_weighted_pool3d(input, irreps, kernel_size, stride):

    assert input.shape[1] == irreps.dim, "Shape mismatch"
    cat_list = []

    start = 0
    for i in irreps.ls:

        end = start + 2*i+1
        temp = input[:,start:end,...]
        if i == 0:
            pooled = F.avg_pool3d(temp, kernel_size, stride=stride)
            cat_list.append(pooled)
        else:
            #breakpoint()
            weights = temp.norm(dim=1).unsqueeze(1)
            weights = torch.ones_like(weights)
            weights = weights / (weights.sum(dim=(2,3,4), keepdims=True) + 1e-8)
            temp_norm = torch.nn.functional.normalize(temp, dim=1)
            pooled = torch.sum(weights*temp_norm, dim=(2,3,4), keepdims=True)
            #pooled = F.avg_pool3d(weights*temp, kernel_size, stride=kernel_size)
            cat_list.append(pooled)
        start = end
    return torch.cat(tuple(cat_list),dim = 1)

def soft_one_hot_linspace(x: torch.Tensor, start, end, number, basis=None, cutoff=None):
    r"""Projection on a basis of functions

    Returns a set of :math:`\{y_i(x)\}_{i=1}^N`,

    .. math::

        y_i(x) = \frac{1}{Z} f_i(x)

    where :math:`x` is the input and :math:`f_i` is the ith basis function.
    :math:`Z` is a constant defined (if possible) such that,

    .. math::

        \langle \sum_{i=1}^N y_i(x)^2 \rangle_x \approx 1

    See the last plot below.
    Note that ``bessel`` basis cannot be normalized.

    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape :math:`(...)`

    start : float
        minimum value span by the basis

    end : float
        maximum value span by the basis

    number : int
        number of basis functions :math:`N`

    basis : {'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}
        choice of basis family; note that due to the :math:`1/x` term, ``bessel`` basis does not satisfy the normalization of other basis choices

    cutoff : bool, string
        if ``cutoff=True`` then for all :math:`x` outside of the interval defined by ``(start, end)``, :math:`\forall i, \; f_i(x) \approx 0`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., N)`

    Examples
    --------

    .. jupyter-execute::
        :hide-code:

        import torch
        from e3nn.math import soft_one_hot_linspace
        import matplotlib.pyplot as plt

    .. jupyter-execute::

        bases = ['gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel']
        x = torch.linspace(-1.0, 2.0, 100)

    .. jupyter-execute::

        fig, axss = plt.subplots(len(bases), 2, figsize=(9, 6), sharex=True, sharey=True)

        for axs, b in zip(axss, bases):
            for ax, c in zip(axs, [True, False]):
                plt.sca(ax)
                plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, number=4, basis=b, cutoff=c))
                plt.plot([-0.5]*2, [-2, 2], 'k-.')
                plt.plot([1.5]*2, [-2, 2], 'k-.')
                plt.title(f"{b}" + (" with cutoff" if c else ""))

        plt.ylim(-1, 1.5)
        plt.tight_layout()

    .. jupyter-execute::

        fig, axss = plt.subplots(len(bases), 2, figsize=(9, 6), sharex=True, sharey=True)

        for axs, b in zip(axss, bases):
            for ax, c in zip(axs, [True, False]):
                plt.sca(ax)
                plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, number=4, basis=b, cutoff=c).pow(2).sum(1))
                plt.plot([-0.5]*2, [-2, 2], 'k-.')
                plt.plot([1.5]*2, [-2, 2], 'k-.')
                plt.title(f"{b}" + (" with cutoff" if c else ""))

        plt.ylim(0, 2)
        plt.tight_layout()
    """
    # pylint: disable=misplaced-comparison-constant

    if cutoff not in [True, False,'left','right']:
        raise ValueError("cutoff must be specified: True, False, 'left', 'right'")

    if cutoff == False:
        values = torch.linspace(start, end, number, dtype=x.dtype, device=x.device)
        step = values[1] - values[0] # [0, 0.625, 1.25, 1.875, 2.5]
    elif cutoff == 'left':
        values = torch.linspace(start, end, number + 1, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[1:]
    elif cutoff == 'right':
        values = torch.linspace(start, end, number + 1, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[:-1]
    else: #cutoff == True
        values = torch.linspace(start, end, number + 2, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[1:-1]

    diff = (x[..., None] - values) / step # shape [5,5,5,5]

    if basis == 'gaussian':
        return diff.pow(2).neg().exp().div(1.12)

    if basis == 'cosine':
        return torch.cos(math.pi/2 * diff) * (diff < 1) * (-1 < diff)

    if basis == 'smooth_finite':
        output = 1.14136 * torch.exp(torch.tensor(2.0)) * soft_unit_step(diff + 1) * soft_unit_step(1 - diff)
        return output

    if basis == 'fourier':
        x = (x[..., None] - start) / (end - start)
        if cutoff == False:
            i = torch.arange(0, number, dtype=x.dtype, device=x.device)
            return torch.cos(math.pi * i * x) / math.sqrt(0.25 + number / 2)
        elif cutoff == 'left':
            i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
            return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (0 < x) 
        elif cutoff == 'right':
            i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
            return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (x < 1)
        else: #cutoff == True
            i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
            return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (0 < x) * (x < 1)

    if basis == 'bessel':
        x = x[..., None] - start
        c = end - start
        bessel_roots = torch.arange(1, number + 1, dtype=x.dtype, device=x.device) * math.pi
        out = math.sqrt(2 / c) * torch.sin(bessel_roots * x / c) / x

        if cutoff == False:
            return out
        elif cutoff == 'left':
            return out * (0 < x)
        elif cutoff == 'right':
            return out * ((x / c) < 1) 
        else:
            return out * ((x / c) < 1) * (0 < x)

    raise ValueError(f"basis=\"{basis}\" is not a valid entry")
