import mindspore.nn as nn
import mindspore.ops.operations as ops


class ReconstructionLoss(nn.Cell):

    def __init__(self, losstype='l2', eps=1e-6):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

        self.mean = ops.ReduceMean()
        self.sum = ops.ReduceSum()
        self.sqrt = ops.Sqrt()

    def construct(self, x, target):
        if self.losstype == 'l2':
            return self.mean(self.sum((x - target)**2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return self.mean(self.sum(self.sqrt(diff * diff + self.eps), (1, 2, 3)))
        else:
            print("reconstruction loss type error!")
            return 0
