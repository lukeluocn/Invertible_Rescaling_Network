# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""define loss function for network"""
# from mindspore.nn.loss.loss import Loss
import mindspore as ms
from mindspore.ops import operations as ops
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.nn as nn

import math
import numpy as np

from models.modules.loss import ReconstructionLoss
from models.modules.Inv_arch import HaarDownsampling, InvBlockExp
from models.modules.block import DenseBlock

class DownNet(nn.Cell):
    def __init__(self, channel_in=3, channel_out=3,subnet_constructor= None,block_num=[], down_num=2):
        super(DownNet, self).__init__()
        self.down_num = down_num
        operations = []
        current_channel = channel_in
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)
                break
            break
        # 一层下采样HaarDownsampling
        # 加上一层InvBlockExp中5层Conv,也即只训练self.F网络
        self.operations = nn.CellList(operations)
        self.conv2 = nn.Conv2d(3,4,(2,2),stride=2,has_bias=True,weight_init='zeros')

    def construct(self, x, rev=False):
        out = x
        print(out.shape)
        for op in self.operations:
            out = op.construct(out, rev)
            print(out.shape)
        out = self.conv2(out)
        return out


def define_G(opt):
    opt_net = opt['network_G']

    down_num = int(math.log(opt_net['scale'], 2))

    netG = DownNet(opt_net['in_nc'], opt_net['out_nc'], DenseBlock, opt_net['block_num'], down_num)

    return netG






class IRN_loss(nn.Cell):
    """the irn network with redefined loss function"""

    def __init__(self, net_G, opt):
        super(IRN_loss, self).__init__()
        self.netG = net_G

        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.img_visual = {}
        self.old_output = []

        self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
        self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

        self.ms_sum = ops.ReduceSum()
        self.cat = ops.Concat(axis=1)
        self.reshape = ops.Reshape()
        self.round = ms.ops.Round()
        self.cast = P.Cast()


    def gaussian_batch(self, dims):
        stdnormal = ops.StandardNormal(self.train_opt['manual_seed'])
        return self.cast(stdnormal(tuple(dims)),ms.float32)

    def loss_forward(self, out, y, z):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)

        z = self.reshape(z, (out.shape[0], -1))
        l_forw_ce = self.train_opt['lambda_ce_forw'] * self.ms_sum(z**2) / z.shape[0]

        return l_forw_fit, l_forw_ce

    def loss_backward(self, x, y):
        x_samples = self.netG(x=y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)

        return l_back_rec

    def construct(self, ref_L, real_H):
        self.input = real_H

        self.output = self.netG(x=self.input)
        print(self.output.shape)
        zshape = self.output[:, 3:, :, :].shape
        LR_ref = ref_L

        l_forw_fit, l_forw_ce = self.loss_forward(self.output[:, :3, :, :], LR_ref, self.output[:, 3:, :, :])

        # total loss
        loss = l_forw_fit + l_forw_ce
        print(loss, l_forw_fit, l_forw_ce) 
        return loss

    def test(self,ref_L, real_H):
    	pass









