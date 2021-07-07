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
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.nn as nn

import numpy as np

from models.modules.loss import ReconstructionLoss

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
        self.cast = ms.ops.Cast()


    def gaussian_batch(self, dims):
        stdnormal = ops.StandardNormal(self.train_opt['manual_seed'])
        return stdnormal(tuple(dims))

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

        new_params = self.netG.trainable_params()
        self.output = self.netG(x=self.input)
        zshape = self.output[:, 3:, :, :].shape
        LR_ref = ref_L

        l_forw_fit, l_forw_ce = self.loss_forward(self.output[:, :3, :, :], LR_ref, self.output[:, 3:, :, :])

        # backward upscaling
        quant = C.clip_by_value(self.output[:, :3, :, :], 0, 1)

        # Quantization of numpy version
        quant = quant.asnumpy()
        LR = (quant * 255.).round() /255.
        LR = Tensor(LR)

        gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] != None else 1
        T = gaussian_scale * self.gaussian_batch(zshape)
        y_ = self.cat((LR, T))

        l_back_rec = self.loss_backward(real_H, y_)

        # total loss
        loss = l_forw_fit + l_back_rec + l_forw_ce
        print(loss, l_forw_fit , l_back_rec , l_forw_ce)
        return loss

    def test(self,ref_L, real_H):
        Lshape = ref_L.shape
        input_dim = Lshape[1]

        self.input = real_H
        self.output = self.netG(x=self.input)

        zshape = [Lshape[0], input_dim * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]

        quant = C.clip_by_value(self.output[:, :3, :, :], 0, 1)

        quant = quant.asnumpy()
        LR = (quant * 255.).round() /255.
        LR = Tensor(LR)

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        T = gaussian_scale * self.gaussian_batch(zshape)
        y_forw = self.cat((LR, T))

        self.fake_H = self.netG(x=y_forw, rev=True)[:, :3, :, :]

        self.img_visual["GT"] = real_H[0]
        self.img_visual['LR_ref'] = ref_L[0]
        self.img_visual["SR"] = self.fake_H[0]
        self.img_visual['LR'] = LR[0]










