import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def define_G(input_dim=3, output_dim=3, ndf=32):

    net_decoder = decoder(output_dim, ndf)
    net_encoder_nir = encoder(input_dim, ndf, get_num_adain_params(net_decoder))
    net_encoder_vis = encoder(input_dim, ndf, get_num_adain_params(net_decoder))

    net_decoder = torch.nn.DataParallel(net_decoder).cuda()
    net_encoder_nir = torch.nn.DataParallel(net_encoder_nir).cuda()
    net_encoder_vis = torch.nn.DataParallel(net_encoder_vis).cuda()

    return net_encoder_nir, net_encoder_vis, net_decoder


class encoder(nn.Module):
    def __init__(self, input_dim, ndf=32, h_dim=256):
        super(encoder, self).__init__()

        self.conv = nn.Sequential(
            convblock(input_dim, ndf, 5, 1, 2),   # 128
            convblock(ndf, 2 * ndf, 3, 2, 1),     # 64
            convblock(2 * ndf, 4 * ndf, 3, 2, 1), # 32
            convblock(4 * ndf, 8 * ndf, 3, 2, 1), # 16
            convblock(8 * ndf, 8 * ndf, 3, 2, 1), # 8
            convblock(8 * ndf, 8 * ndf, 3, 2, 1)  # 4
        )

        self.fc_enc = nn.Linear(8 * ndf * 4 * 4, 256)

        self.fc_style = nn.Sequential(
            nn.Linear(256, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, h_dim)
        )

    def forward(self, x, state='enc'):
        if state == 'style':
            x = F.normalize(x, p=2, dim=1)
            style = self.fc_style(x)
            return style

        elif state == 'enc':
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc_enc(x)
            return F.normalize(x, p=2, dim=1)


class decoder(nn.Module):
    def __init__(self, output_dim=3, ndf=32):
        super(decoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(256+256, 4 * ndf * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv = nn.Sequential(
            deconvblock(4 * ndf, 4 * ndf, 2, 2, 0),  # 8
            resblock(4 * ndf, 4 * ndf),
            deconvblock(4 * ndf, 4 * ndf, 2, 2, 0),  # 16
            resblock(4 * ndf, 4 * ndf),
            deconvblock(4 * ndf, 2 * ndf, 2, 2, 0),  # 32
            resblock(2 * ndf, 2 * ndf),
            deconvblock(2 * ndf, 2 * ndf, 2, 2, 0),  # 64
            resblock(2 * ndf, 2 * ndf),
            deconvblock(2 * ndf, ndf, 2, 2, 0, norm='adain'),  # 128
            resblock(ndf, ndf, norm='adain'),
            convblock(ndf, ndf, 3, 1, 1, norm='adain'),
            resblock(ndf, ndf, norm='adain'),
            convblock(ndf, ndf, 3, 1, 1, norm='adain')
        )

        self.nir_output = nn.Conv2d(ndf, output_dim, 1, 1, 0)
        self.vis_output = nn.Conv2d(ndf, output_dim, 1, 1, 0)

    def forward(self, x, modality='nir'):
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.conv(x)

        if modality == 'nir':
            x = self.nir_output(x)
        elif modality == 'vis':
            x = self.vis_output(x)
        return torch.sigmoid(x)


# basic module
class resblock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='in'):
        super(resblock, self).__init__()

        self.conv1 = convblock(input_dim, output_dim, 3, 1, 1, norm)
        self.conv2 = convblock(output_dim, output_dim, 3, 1, 1, norm)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y

class convblock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, norm='in'):
        super(convblock, self).__init__()

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(output_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(output_dim)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class deconvblock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, norm='in'):
        super(deconvblock, self).__init__()

        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(output_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(output_dim)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

# for AdaIN
def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
