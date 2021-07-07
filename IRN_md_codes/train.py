
import os
import math
import argparse
import random
import numpy as np
import logging
import ast

import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common import set_seed
from mindspore.nn import layer
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.model import ParallelMode
from mindspore import Model,load_checkpoint,save_checkpoint,load_param_into_net

import options.options as option
from utils import util
from models import create_model,TrainOneStepCell_IRN,print_network
from models.irn_loss import IRN_loss
from data import create_dataset
from models.warmup_multisteplr import warmup_step_lr
from models.warmup_cosine_annealing_lr import warmup_cosine_annealing_lr
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="srcnn training")
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--device_num', type=int, default=1, help='Device num.')
    parser.add_argument('--device_target', type=str, default='GPU', choices=("GPU"),
                        help="Device target, support GPU.")
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default: false.")  # 对字符串进行类型转换的同时兼顾系统的安全考虑

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    ### 设置pynative mode
    if args.device_target == "GPU":
        context.set_context(mode=context.PYNATIVE_MODE,
                            device_target=args.device_target,
                            save_graphs=False)
    else:
        raise ValueError("Unsupported device target.")

    rank = 0
    device_num = args.device_num
    if args.run_distribute:
        opt['dist'] = True
        init()
        rank = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL)
    else:
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')

    opt = option.dict_to_nonedict(opt)
    
    ### 得到训练集和验证集
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_dataset = create_dataset(phase, dataset_opt,opt['gpu_ids'])
        elif phase == 'val':
            val_loader = create_dataset(phase, dataset_opt,opt['gpu_ids'])

    dataset_opt = opt['datasets']['train']
    train_dataset = create_dataset("train",dataset_opt,opt['gpu_ids'])

    # train_size = int(math.ceil(len(train_dataset) / dataset_opt['batch_size']))
    step_size = train_dataset.get_dataset_size()
    total_iters = int(opt['train']['niter'])
    total_epochs = int(math.ceil(total_iters / step_size))
    
    print("Total epoches : {}".format(total_epochs))


    train_opt = opt['train']
    test_opt = opt['test']

    ### 创建日志信息路径
    util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,screen=True, tofile=True)
    util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO, screen=True, tofile=True)


    ### 得到学习率
    wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
    if train_opt['lr_scheme'] == 'MultiStepLR':
        lr = warmup_step_lr(train_opt['lr_G'],
                            train_opt['lr_steps'],
                            step_size,
                            0,
                            total_epochs,
                            train_opt['lr_gamma'],
                            )
    elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
        lr = warmup_cosine_annealing_lr(train_opt['lr_G'],
                                        train_opt['lr_steps'],
                                        0,
                                        total_epochs,
                                        train_opt['restarts'],
                                        train_opt['eta_min'])
    else:
        raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

    #### define net    
    net = create_model(opt)
    # print_network(net,"IRN")

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        param_dict = load_checkpoint(opt['path']['resume_state'])
        # if args.filter_weight:
        #     filter_list = [x.name for x in net.end_point.get_parameters()]
        #     filter_checkpoint_parameter_by_list(param_dict, filter_list)
        load_param_into_net(net, param_dict)

    #### define network with loss
    loss = IRN_loss(net,opt) 

    #### warp network with optimizer
    optimizer = nn.Adam(loss.netG.trainable_params(), learning_rate=Tensor(lr),
                 beta1=train_opt['beta1'], beta2=train_opt['beta2'], weight_decay=wd_G)
    model = TrainOneStepCell_IRN(loss, optimizer)

    #### set train
    model.set_train()

    """  使用train的整体训练API
    #### 1. Warp the whole network as a Model
    # irn_model = Model(model)
    # irn_model = Model(network=loss,optimizer=optimizer)

    ### 2. define callbacks
    # callbacks = [LossMonitor(), TimeMonitor(data_size=step_size)]
    # config_ck = CheckpointConfig(save_checkpoint_steps=step_size,
    #                              keep_checkpoint_max=30)
    # save_ckpt_path = os.path.join('ckpt/', 'ckpt_' + str(rank) + '/')
    # ckpt_cb = ModelCheckpoint(prefix="irn", directory=save_ckpt_path, config=config_ck)
    # callbacks.append(ckpt_cb)

    ## 注释语句
    # initial_param = loss.netG.trainable_params()
    # i_param = []
    # for i in range(len(initial_param)):
    #     i_param.append(initial_param[i].asnumpy().copy())
    # i_param = np.array(i_param.copy())


    ### 3. train 
    # total_epochs = 5
    # irn_model.train(total_epochs, train_dataset, callbacks=callbacks, dataset_sink_mode=False)

    ## 注释语句，用于判断训练前后模型参数是否发生变化
    # new_params = irn_model._network.netG.trainable_params()
    # for j in range(len(initial_param)):
    #     if (i_param[j] == new_params[j].asnumpy()).all():
    #         print(initial_param[j], "not changed at all")
    #     else:
    #         print(initial_param[j], "changed")
    """

    print(lr.shape,total_epochs)
    # (500000,) 2500
    
    ### 单batch逐步进行训练
    dataset_iter = train_dataset.create_dict_iterator()
    data = next(dataset_iter)
    lq = ms.Tensor(data["low_quality"],ms.float32)
    gt = ms.Tensor(data["ground_truth"],ms.float32)      

    ## 得到训练前的模型参数
    initial_param = net.trainable_params()
    i_param = []
    for i in range(len(initial_param)):
        i_param.append(initial_param[i].asnumpy().copy())
    i_param = np.array(i_param.copy())
    layer_num = len(initial_param)
    print(layer_num)
    
    for i in range(total_epochs):
        loss1 = model(lq,gt)  
        print("======",i," epochs get the loss as", loss1)
        print("the learning rate is ", lr[i])

        new_params = model.G.netG.trainable_params() # 已更新的新参数

        ### 输出信息语句，用以判断训练前后模型参数是否更新
        f=open("parame.txt","w")
        for j in range(layer_num):
            if (i_param[j] == new_params[j].asnumpy()).all():
                print(initial_param[j], "not changed at all")
            else:
                print(initial_param[j], "changed")
            f.write(str(initial_param[j])+str(new_params[j].asnumpy().shape)+"\n")
        f.close()


        if i==1:
            break    ## 节约时间仅跑一轮

