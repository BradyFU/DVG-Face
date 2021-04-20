import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def define_ID():
    net_dec = dec()
    net_dec = torch.nn.DataParallel(net_dec).cuda()
    return net_dec


class dec(nn.Module):
    def __init__(self):
        super(dec, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(256, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)
