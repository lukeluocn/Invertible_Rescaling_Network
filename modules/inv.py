from IRN_md_codes.simplified_models.modules.block import DenseBlock
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as ops


class InvBlockExp(nn.Cell):

    def __init__(self, block, channel_num, channel_split_num):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num
        self.F = block(self.split_len2, self.split_len1)

    def construct(self, x, rev=False):
        x1 = x[:, 0 : 0 + self.split_len1]
        x2 = x[:, self.split_len1 : self.split_len1 + self.split_len2]
        if not rev:
            y1 = x1 + self.F(x2)
        return y1


class HaarDownsampling(nn.Cell):

    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        haar_weights = np.ones((4, 1, 2, 2))

        haar_weights[1, 0, 0, 1] = -1
        haar_weights[1, 0, 1, 1] = -1

        haar_weights[2, 0, 1, 0] = -1
        haar_weights[2, 0, 1, 1] = -1

        haar_weights[3, 0, 1, 0] = -1
        haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = np.concatenate([haar_weights] * self.channel_in, 0).astype(np.float32)
        self.haar_weights = ms.Tensor(self.haar_weights)
        self.haar_weights.requires_grad = False

        self.conv2d = ms.ops.Conv2D(
            out_channel=self.haar_weights.shape[0],
            kernel_size=(self.haar_weights.shape[2], self.haar_weights.shape[3]),
            stride=2,
            group=self.channel_in,
        )

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

        self.conv2d_transpose = ms.ops.Conv2DBackpropInput(
            out_channel=self.haar_weights.shape[1] * self.channel_in,
            kernel_size=(self.haar_weights.shape[2], self.haar_weights.shape[3]),
            stride=2,
            group=self.channel_in,
        )

    def construct(self, x, rev=False):
        if not rev:
            out = self.conv2d(x, self.haar_weights) / 4.0
            out = self.reshape(
                out,
                (x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2),
            )
            out = self.transpose(out, (0, 2, 1, 3, 4))
            out = self.reshape(
                out,
                (x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2),
            )
            return out

        out = self.reshape(x, (x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]))
        out = self.transpose(out, (0, 2, 1, 3, 4))
        out = self.reshape(
            out,
            (x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]),
        )
        return self.conv2d_transpose(
            out,
            self.haar_weights,
            (out.shape[0], self.haar_weights.shape[1] * self.channel_in,
             out.shape[2] * 2, out.shape[3] * 2),
        )


class InvRescaleNet(nn.Cell):

    def __init__(self, channel_in=3, channel_out=3, block=DenseBlock, block_num=[], down_num=2):
        super(InvRescaleNet, self).__init__()

        current_channel = channel_in
        operations = []
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for _ in range(block_num[i]):
                b = InvBlockExp(block, current_channel, channel_out)
                operations.append(b)
        self.operations = nn.CellList(operations)

    def construct(self, x, rev=False):
        out = x

        if not rev:
            for op in self.operations:
                out = op.construct(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.construct(out, rev)

        return out
