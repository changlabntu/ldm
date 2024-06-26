# Copyright 2020 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.model_utils import get_activation, Identity

sig = nn.Sigmoid()
ACTIVATION = nn.ReLU
#device = 'cuda'


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.reshape(x.size()[0], -1)

def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))

    return torch.cat((upsampled, bypass), 1)


def conv2d_bn_block(in_channels, out_channels, kernel=3, momentum=0.01, activation=ACTIVATION):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=1),
        nn.BatchNorm2d(out_channels, momentum=momentum),
        activation(),
    )


def deconv2d_bn_block(in_channels, out_channels, use_upsample=True, kernel=4, stride=2, padding=1, momentum=0.01,
                      activation=ACTIVATION):
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        nn.BatchNorm2d(out_channels, momentum=momentum),
        activation(),
    )


def dense_layer_bn(in_dim, out_dim, momentum=0.01, activation=ACTIVATION):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim, momentum=momentum),
        activation()
    )


def conv2d_block(in_channels, out_channels, kernel=3, stride=1, padding=1, activation=ACTIVATION):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )


class Generator(nn.Module):
    def __init__(self, n_channels=1, out_channels=1, nf=32, batch_norm=True, activation=ACTIVATION, final='tanh', mc=False):
        super(Generator, self).__init__()

        conv_block = conv2d_bn_block if batch_norm else conv2d_block

        max_pool = nn.MaxPool2d(2)
        act = activation
        self.label_k = torch.tensor([0, 1]).half()
        self.c_dim = 0

        self.down0 = nn.Sequential(
            conv_block(n_channels + self.c_dim, nf, activation=act),
            conv_block(nf, nf, activation=act)
        )
        self.down1 = nn.Sequential(
            max_pool,
            conv_block(nf, 2 * nf, activation=act),
            conv_block(2 * nf, 2 * nf, activation=act),
        )
        self.down2 = nn.Sequential(
            max_pool,
            conv_block(2 * nf, 4 * nf, activation=act),
            conv_block(4 * nf, 4 * nf, activation=act),

        )
        self.down3 = nn.Sequential(
            max_pool,
            conv_block(4 * nf, 8 * nf, activation=act),
            conv_block(8 * nf, 8 * nf, activation=act),
        )

        self.up3 = deconv2d_bn_block(8 * nf, 4 * nf, activation=act)

        self.conv5 = nn.Sequential(
            conv_block(8 * nf, 4 * nf, activation=act),  # 8
            conv_block(4 * nf, 4 * nf, activation=act),
        )
        self.up2 = deconv2d_bn_block(4 * nf, 2 * nf, activation=act)
        self.conv6 = nn.Sequential(
            conv_block(4 * nf, 2 * nf, activation=act),
            conv_block(2 * nf, 2 * nf, activation=act),
        )

        self.up1 = deconv2d_bn_block(2 * nf, nf, activation=act)

        final_layer = get_activation(final)

        self.conv7_k = nn.Sequential(
            conv_block(nf, nf, activation=act),
            conv_block(nf, out_channels, activation=final_layer),
        )

        self.conv7_g = nn.Sequential(
            conv_block(nf, nf, activation=act),
            conv_block(nf, out_channels, activation=final_layer),
        )

        #if NoTanh:
        #    self.conv7_k[-1] = self.conv7_k[-1][:-1]
        #    self.conv7_g[-1] = self.conv7_g[-1][:-1]

    def forward(self, xori, a=None):
        x = 1 * xori
        # c: (B, C)
        self.c_dim = 0
        if self.c_dim > 0:
            c = a
            c1 = c.view(c.size(0), c.size(1), 1, 1)
            c1 = c1.repeat(1, 1, x.size(2), x.size(3))  # (B, 2, H, W)
            x = torch.cat([x, c1], dim=1)

        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        xu3 = self.up3(x3)
        cat3 = torch.cat([xu3, x2], 1)
        x5 = self.conv5(cat3)
        xu2 = self.up2(x5)
        cat2 = torch.cat([xu2, x1],1)

        x6 = self.conv6(cat2)
        xu1 = self.up1(x6)
        #cat1 = crop_and_concat(xu1, x0)

        #if self.label_k in c:
        x70 = self.conv7_k(xu1)
        #else:
        x71 = self.conv7_g(xu1)

        return {'out0': x70, 'out1': x71,'z': x3}  # return x70, x71


class Discriminator(nn.Module):

    def __init__(self, n_channels=6, init_filters=16, batch_norm=False):
        super(Discriminator, self).__init__()
        nf = init_filters
        self.label_k = torch.ones(1).long()

        conv_block = conv2d_bn_block if batch_norm else conv2d_block

        max_pool = nn.MaxPool2d
        self.encoder = nn.Sequential(
            conv_block(n_channels, nf),
            max_pool(2),
            conv_block(nf, 2 * nf),

            max_pool(2),
            conv_block(2 * nf, 4 * nf),
            conv_block(4 * nf, 4 * nf),
            max_pool(2),
            conv_block(4 * nf, 8 * nf),
            conv_block(8 * nf, 8 * nf),
            max_pool(2),
            conv_block(8 * nf, 8 * nf),
            conv_block(8 * nf, 8 * nf),
            max_pool(2),
            conv_block(8 * nf, 16 * nf),

        )
        kernel_size = int(240 / np.power(2, 5))
        self.conv_k = nn.Sequential(conv_block(16 * nf, 16 * nf), conv_block(16 * nf, 16 * nf),
                                    conv_block(16 * nf, 1, kernel=1, activation=Identity), )
        self.conv_g = nn.Sequential(conv_block(16 * nf, 16 * nf), conv_block(16 * nf, 16 * nf),
                                    conv_block(16 * nf, 1, kernel=1, activation=Identity), )
        self.conv2 = nn.Sequential(conv_block(16 * nf, 16 * nf), conv_block(16 * nf, 16 * nf), max_pool(2), )

        self.linearclass = nn.Sequential(

            Flatten(),
            nn.Linear(16 * nf * 4 * 4, 64),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 2),

        )

    def forward(self, x, label=None):
        h = self.encoder(x)
        out0 = self.conv_k(h)
        out1 = self.conv_g(h)
        return out0, h #out1


if __name__ == '__main__':
    g = Generator(n_channels=3, batch_norm=False, final='tanh')
    #from torchsummary import summary
    from utils.data_utils import print_num_of_parameters
    print_num_of_parameters(g)

    d = Discriminator()
    dout = d(torch.rand(1, 6, 256, 256))
    #summary(g, [(3, 256, 256), (2)])
    #o = g(torch.rand(2, 3, 256, 256).cuda(), torch.ones(2, 2).cuda())