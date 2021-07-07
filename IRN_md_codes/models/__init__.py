import logging
from mindspore.ops.composite import clip_ops
logger = logging.getLogger('base')
import numpy as np

import models.networks as networks
from models.modules.loss import ReconstructionLoss
from models.irn_loss import IRN_loss

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as ops
from mindspore import dtype as mstype
from mindspore.train.serialization import load_checkpoint,load_param_into_net


def print_network(model,name):
    num_params=0
    for p in model.trainable_params():
        num_params += np.prod(p.shape)
    print(model)
    print(name)
    print('the number of parameters : {}'.format(num_params))

def create_model(opt):
    model = opt['model']
    if model == 'IRN':
        # m = M(opt)
        m = networks.define_G(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m


"""
    Warp the network with loss function to return generator loss
"""
class NetWithLossCell(nn.Cell):
    def __init__(self, network):
        super(NetWithLossCell,self).__init__(auto_prefix=False)
        self.network = network
    
    def construct(self, ref_L, real_H ):
        loss = self.network(ref_L, real_H )
        return loss


class TrainOneStepCell_IRN(nn.Cell):
    """Encapsulation class of IRN network training
    
        Appending an optimizer to the training network after that the construct function can be called to create the backward graph.
    """
    def __init__(self,G,optimizer,sens=1.0):
        super(TrainOneStepCell_IRN,self).__init__()
        self.optimizer = optimizer
        self.G = G
        self.G.set_grad()
        self.G.set_train(True)
        self.grad = ms.ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.reducer_flag = False
        # self.network = NetWithLossCell(G)
        self.G.add_flags(defer_inline=True)
        self.grad_reducer=F.identity
        self.image_visuals = {}

        self.stack = ms.ops.Stack()
        self.norm = nn.Norm()
        self.mul = ms.ops.Mul()

    ### 测试模型输出
    def test(self, ref_L, real_H ):
        self.G.set_train(False)
        self.G.test(ref_L, real_H )
        self.G.set_train(True)
        self.image_visuals = self.G.img_visual
        
        return self.image_visuals

    def construct(self,ref_L, real_H ):
        self.G.set_train(True)

        ### get the gradient
        loss = self.G(ref_L, real_H )
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.G, self.weights)(ref_L, real_H , sens) 
        

        ### clipping gradient norm
        max_norm = float(10)
        total_norm = 0.0
        norm_type = 2.0
        for grad in grads:
            param_norm = self.norm(grad)
            total_norm += param_norm**norm_type
        total_norm = total_norm ** (1. / norm_type)
        # total_norm = self.norm(self.stack([self.norm(grad) for grad in grads]))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for grad in grads:
                grad.set_data(self.mul(grad,clip_coef)) ## 更新梯度

        grads = self.grad_reducer(grads) 
        ### 打印相关内容
        # print("grad write into the txt file")
        # with open("grad.txt","w") as f:
        #     f.write(str(grads)) 

        return F.depend(loss,self.optimizer(grads))
