# Invertible_Rescaling_Network
Mindspore version for IRN


## 论文
Invertible Image Rescaling

## 参考实现
https://github.com/pkuxmq/Invertible-Image-Rescaling


## 代码目录结构
```
├── data
│   ├── __init__.py
│   ├── LQGT_dataset.py
│   ├── test.png
│   └── util.py
├── models    # 模型网络代码
│   ├── __init__.py
│   ├── irn_loss.py
│   ├── modules     # 网络组件部分
│   │   ├── Inv_arch.py
│   │   ├── loss.py
│   │   ├── module_util.py
│   │   └── Subnet_constructor.py
│   ├── networks.py
│   ├── warmup_cosine_annealing_lr.py
│   └── warmup_multisteplr.py
├── options
│   ├── options.py
│   ├── test
│   │   ├── test_IRN_x2.yml
│   │   ├── test_IRN+_x4.yml
│   │   └── test_IRN_x4.yml
│   └── train
│       ├── train_IRN_x2.yml
│       ├── train_IRN+_x4.yml
│       └── train_IRN_x4.yml
├── run_scripts.sh
├── simplified_models   # 简化网络框架
│   ├── DIY_test_loss.py   # 自定义网络  only for debugging
│   ├── __init__.py
│   ├── modules
│   │   ├── Inv_arch.py
│   │   ├── loss.py
│   │   ├── module_util.py
│   │   └── Subnet_constructor.py
│   ├── warmup_cosine_annealing_lr.py
│   └── warmup_multisteplr.py
├── test_dp.py
├── train.py    # 训练文件
└── utils
    └── util.py

```
