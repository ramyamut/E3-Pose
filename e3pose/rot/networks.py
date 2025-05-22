import torch
import torch.nn as nn

from .e3cnn import ConvNet

class E3CNN_Encoder(nn.Module):
    """
    Wrapper class for E3CNN encoder architecture
    """

    def __init__(self, input_chans, output_chans, n_levels, k, last_activation=None, return_fmaps=False, equivariance='O3'):
        super(E3CNN_Encoder, self).__init__()
        
        if last_activation == 'relu':
            self.last_activation = nn.ReLU()
        elif last_activation == 'elu':
            self.last_activation = nn.ELU()
        elif last_activation == 'softmax':
            self.last_activation = nn.Softmax()
        elif last_activation == 'tanh':
            self.last_activation = nn.Tanh()
        elif last_activation == 'sigmoid':
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None
        self.return_fmaps = return_fmaps
        self.net = ConvNet(f'{input_chans}x0e',f'{output_chans}x1e + {output_chans*2}x1o',k,k,(1,1,1),n_downsample=n_levels,equivariance=equivariance, lmax=4, return_fmaps=self.return_fmaps) 

    def forward(self, x, pool=True):
        x = self.net.forward(x.to(dtype=torch.float32), pool=pool)
        if self.last_activation is not None:
            if self.return_fmaps:
                x[-1] = self.last_activation(x[-1])
            else:
                x = self.last_activation(x)
        return x
    
    def pool(self, x):
        return self.net.pool(x)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        return self

