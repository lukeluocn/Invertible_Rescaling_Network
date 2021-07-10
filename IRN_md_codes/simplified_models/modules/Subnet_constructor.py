# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import models.modules.module_util as mutil

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import operations as ops

import models.modules.module_util as mutil
# 

class DenseBlock(nn.Cell):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        # self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        # self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        # self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        # self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        # self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv1 = nn.Conv2d(channel_in         , gc, 3, 1,'pad',1, has_bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc    , gc, 3, 1,'pad',1, has_bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1,'pad',1, has_bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1,'pad',1, has_bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1,'pad',1, has_bias=bias)
        self.lrelu = nn.LeakyReLU(alpha=0.2)  
        self.cat = ops.Concat(axis=1)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def construct(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(self.cat((x, x1))))
        x3 = self.lrelu(self.conv3(self.cat((x, x1, x2))))
        x4 = self.lrelu(self.conv4(self.cat((x, x1, x2, x3))))
        x5 = self.conv5(self.cat((x, x1, x2, x3, x4)))

        return x5


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        else:
            return None

    return constructor
