import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import operations as ops
from mindspore import dtype as mstype


class InvBlockExp(nn.Cell):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)

        self.sigmoid = ops.Sigmoid()
        self.exp = ops.Exp()
        self.cat = ops.Concat(axis=1)
        self.sum = ops.ReduceSum()
        self.mul = ops.Mul()
        self.div = ops.Div()

    def construct(self, x, rev=False):
        x1,x2 = x[:,0:0+self.split_len1],x[:,self.split_len1:self.split_len1+self.split_len2]

        if not rev:
            y1 = x1 + self.F(x2)

        return y1

    def jacobian(self, x, rev=False):
        if not rev:
            jac = self.sum(self.s)
        else:
            jac = -self.sum(self.s)

        return jac / x.shape[0]


class HaarDownsampling(nn.Cell):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = np.ones((4, 1, 2, 2))

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = np.concatenate([self.haar_weights] * self.channel_in, 0).astype(np.float32)
        # self.haar_weights = ms.Parameter(ms.Tensor(self.haar_weights),name='haar_w',requires_grad=False)
        self.haar_weights = ms.Tensor(self.haar_weights)
        self.haar_weights.requires_grad = False

        self.conv2d = ms.ops.Conv2D(out_channel=self.haar_weights.shape[0],
                                kernel_size=(self.haar_weights.shape[2],self.haar_weights.shape[3]),
                                stride=2,
                                group=self.channel_in)

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

        self.conv2d_transpose = ms.ops.Conv2DBackpropInput(out_channel=self.haar_weights.shape[1]*self.channel_in,
                                                    kernel_size=(self.haar_weights.shape[2],self.haar_weights.shape[3]),
                                                    stride=2,
                                                    group=self.channel_in
                                                    )

    def construct(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            # out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = self.conv2d(x,self.haar_weights) / 4.0
            # out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = self.reshape(out,(x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2))
            # out = torch.transpose(out, 1, 2)
            out = self.transpose(out,(0,2,1,3,4))
            # out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            out = self.reshape(out,(x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2))
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            # out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = self.reshape(x,(x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]))
            # out = torch.transpose(out, 1, 2)
            out = self.transpose(out,(0,2,1,3,4))
            # out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            out = self.reshape(out,(x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]))
            # return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)
            # return self.conv2d_transpose(out)
            return self.conv2d_transpose(out,self.haar_weights,(out.shape[0],self.haar_weights.shape[1]*self.channel_in,out.shape[2]*2,out.shape[3]*2))

    def jacobian(self, x, rev=False):
        return self.last_jac 


class InvRescaleNet(nn.Cell):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2):
        super(InvRescaleNet, self).__init__()

        operations = []
        current_channel = channel_in
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)
        self.operations = nn.CellList(operations)

    def construct(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.construct(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.construct(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out
            




