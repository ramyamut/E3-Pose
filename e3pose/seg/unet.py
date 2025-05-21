import torch
from torch import nn

class UNet(nn.Module):

    def __init__(self,
                 n_input_channels=1,
                 n_output_channels=1,
                 n_levels=3,
                 n_conv=2,
                 n_feat=32,
                 feat_mult=1,
                 kernel_size=3,
                 activation='relu',
                 last_activation=None,
                 batch_norm_after_each_conv=False,
                 residual_blocks=False,
                 encoder_only=False,
                 upsample=False,
                 use_skip_connections=True,
                 rm_top_skip_connection=0,
                 predict_residual=False):

        super(UNet, self).__init__()

        # input/output channels
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels

        # general architecture
        self.encoder_only = encoder_only
        self.upsample = upsample
        self.rm_top_skip_connection = rm_top_skip_connection if use_skip_connections else self.n_levels
        self.predict_residual = predict_residual

        # convolution block parameters
        self.n_levels = n_levels
        self.n_conv = n_conv
        self.feat_mult = feat_mult
        self.feat_list = [n_feat * feat_mult ** i for i in range(self.n_levels)]
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_norm_after_each_conv = batch_norm_after_each_conv
        self.residual_blocks = residual_blocks

        # define convolutional blocks
        self.list_encoder_blocks = self.get_list_encoder_blocks()  # list of length self.n_levels
        if not self.encoder_only:
            self.list_decoder_blocks = self.get_list_decoder_blocks()  # list of length self.n_levels - 1
            self.last_conv = torch.nn.Conv3d(self.feat_list[0], self.n_output_channels, kernel_size=1)
        else:
            self.list_decoder_blocks = []
            self.last_conv = torch.nn.Conv3d(self.feat_list[-1], self.n_output_channels, kernel_size=1)

        if last_activation == 'relu':
            self.last_activation = torch.nn.ReLU()
        elif last_activation == 'elu':
            self.last_activation = torch.nn.ELU()
        elif last_activation == 'softmax':
            self.last_activation = torch.nn.Softmax(dim=1)
        elif last_activation == 'tanh':
            self.last_activation = torch.nn.Tanh()
        elif last_activation == 'sigmoid':
            self.last_activation = torch.nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        """takes tuple of two inputs, each with the same shape [B, C, H, W, D]"""

        tens = x

        # down-arm
        list_encoders_features = []
        #breakpoint()
        for i, encoder_block in enumerate(self.list_encoder_blocks):
            if i > 0:
                tens = torch.nn.functional.max_pool3d(tens, kernel_size=2)
            tens_out = encoder_block(tens)
            tens = tens + tens_out if self.residual_blocks else tens_out
            list_encoders_features.append(tens)

        # up-arm
        if not self.encoder_only:

            # remove output of last encoder block (i.e. the bottleneck) from the list of features to be concatenated
            list_encoders_features = list_encoders_features[::-1][1:]

            # build conv
            for i in range(len(self.list_decoder_blocks)):
                tens = torch.nn.functional.upsample(tens, scale_factor=2, mode='trilinear')
                if i < (self.n_levels - 1 - self.rm_top_skip_connection):
                    tens_out = torch.cat((list_encoders_features[i], tens), dim=1)
                else:
                    tens_out = tens
                tens_out = self.list_decoder_blocks[i](tens_out)
                tens = tens + tens_out if self.residual_blocks else tens_out

        # final convolution
        tens = self.last_conv(tens)
        if self.last_activation is not None:
            tens = self.last_activation(tens)

        if self.upsample:
            tens = torch.nn.functional.interpolate(tens, scale_factor=2 ** (self.n_levels - 1), mode='trilinear')

        # residual
        if self.predict_residual:
            tens = x + tens

        return tens

    def get_list_encoder_blocks(self):

        list_encoder_blocks = []
        for i in range(self.n_levels):

            # number of input/output feature maps for each convolution
            if i == 0:
                n_input_feat = [self.n_input_channels] + [self.feat_list[i]] * (self.n_conv - 1)
            else:
                n_input_feat = [self.feat_list[i - 1]] + [self.feat_list[i]] * (self.n_conv - 1)
            n_output_feat = self.feat_list[i]

            # build conv block
            layers = self.build_block(n_input_feat, n_output_feat)
            list_encoder_blocks.append(torch.nn.Sequential(*layers))

        return nn.ModuleList(list_encoder_blocks)

    def get_list_decoder_blocks(self):

        list_decoder_blocks = []
        for i in range(0, self.n_levels - 1):

            # number of input/output feature maps for each convolution
            if i < (self.n_levels - 1 - self.rm_top_skip_connection):
                n_input_feat = [self.feat_list[::-1][i + 1] * (1 + self.feat_mult)] + \
                               [self.feat_list[::-1][i + 1]] * (self.n_conv - 1)
            else:
                n_input_feat = [self.feat_list[::-1][i]] + \
                               [self.feat_list[::-1][i + 1]] * (self.n_conv - 1)
            n_output_feat = self.feat_list[::-1][i + 1]

            # build conv block
            layers = self.build_block(n_input_feat, n_output_feat)
            list_decoder_blocks.append(torch.nn.Sequential(*layers))

        return nn.ModuleList(list_decoder_blocks)

    def build_block(self, n_input_feat, n_output_feat):

        # convolutions + activations
        layers = list()
        for conv in range(self.n_conv):
            layers.append(torch.nn.Conv3d(n_input_feat[conv], n_output_feat, kernel_size=self.kernel_size,
                                          padding=self.kernel_size // 2))
            if self.activation == 'relu':
                layers.append(torch.nn.ReLU())
            elif self.activation == 'elu':
                layers.append(torch.nn.ELU())
            else:
                raise ValueError('activation should be relu or elu, had: %s' % self.activation)
            if self.batch_norm_after_each_conv:
                layers.append(torch.nn.BatchNorm3d(n_output_feat))

        # batch norm
        if not self.batch_norm_after_each_conv:
            layers.append(torch.nn.BatchNorm3d(n_output_feat))

        return layers

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        return self
