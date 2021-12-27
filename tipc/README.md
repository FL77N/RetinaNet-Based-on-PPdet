
# 飞桨训推一体认证

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。
本文档提供了 Retinanet 的飞桨训推一体认证 (Training and Inference Pipeline Certification(TIPC)) 信息和测试工具，轻量化模型的训推一体化以及 serving 端的部署测试。
方便用户查阅每种模型的训练推理及其他部署方法打通情况，并可以进行一键测试。


## 2. 汇总信息

已填写的部分表示可以使用本工具进行一键测试，未填写的表示正在支持中。

**字段说明：**
- 基础训练预测：包括模型训练、混合精度、Paddle Inference Python预测、Serving 预测。
- 更多训练方式：包括多机多卡。

| 算法论文 | 模型名称 | 模型类型 | 基础<br>训练预测 | 更多<br>训练方式 | 模型压缩 |  其他预测部署  |
| :--: | :--- | :----: | :--------: | :---- | :---: | :---: |
| [Retinanet](https://arxiv.org/abs/1708.02002) | [retinanet_r50_fpn_1x_coco](tipc/train_infer_python/configs/retinanet/retinanet_r50_fpn_1x_coco_train_infer_python.txt) | 目标检测  | 支持 | 混合精度 | x | Serving |
|  | [retinanet_mobilenet_v1_fpn_1x_coco](tipc/train_infer_python/configs/retinanet/retinanet_mobilenet_v1_fpn_1x_coco_train_infer_python.txt) | 目标检测<br/>轻量化 | 支持 | 混合精度 | x | Serving |
|  | [retinanet_mobilenet_v3_small_fpn_1x_coco](tipc/train_infer_python/configs/retinanet/retinanet_mobilenet_v3_small_fpn_1x_coco_train_infer_python.txt) | 目标检测<br/>轻量化 | 支持 | 混合精度 | x | Serving |
|  | [retinanet_mobilenet_v3_large_fpn_1x_coco](tipc/train_infer_python/configs/retinanet/retinanet_mobilenet_v3_large_fpn_1x_coco_train_infer_python.txt) | 目标检测<br/>轻量化 | 支持 | 混合精度 | x | Serving |
|  | [retinanet_shufflenet_v2_fpn_1x_coco](tipc/train_infer_python/configs/retinanet/retinanet_shufflenet_v2_fpn_1x_coco_train_infer_python.txt) | 目标检测<br/>轻量化 | 支持 | 混合精度 | x | Serving |


## 3. 测试工具简介
### 目录介绍

```shell
tipc/
├── train_infer_python/                              # 训推测相关文件
│   ├── configs/									 # 相关配置文件
│       ├── retinanet/
│           ├── retinanet_r50_fpn_1x_coco.txt
│           ├── ...
│   ├── output/                                      # 训推测中的 log 及模型保存
│       ├── results_python.log
│       ├── ...
│   ├── weights/                                     # 预训练权重和最优权重
│       ├── xxx.pdparams
│       ├── ...
│   ├── prepare.sh                                   # 完成 test_*.sh 运行所需要的数据和模型下载
│   ├── common_func.sh                               # 共同的工具函数
│   ├── test_train_inference_python.sh               # 测试 python 训练预测的主程序
│   ├── configs/
│   ├── README.md                                    # 训推测使用文档
│
├── serving/                                         # serving 端部署相关文件
│   ├── configs/									 # 相关配置文件
│       ├── retinanet_r50_fpn_1x_coco.txt
│   ├── output/                                      # 训推测中的 log 及模型保存
│       ├── results_serving.log
│       ├── ...
│   ├── test_train_inference_python.sh               # 测试 serving 端预测的主程序
│   ├── README.md                                    # serving 端预测使用文档
│
├── requirements.txt                                 # 需要安装的相关依赖

```

### 测试流程概述

1. 运行prepare.sh准备测试所需数据和模型；
2. 运行要测试的功能对应的测试脚本`test_*.sh`，产出 log，由 log 可以看到不同配置是否运行成功；

测试单项功能仅需两行命令，**如需测试不同模型/功能，替换配置文件即可**，命令格式如下：
```shell
# 功能：准备数据
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/prepare.sh  configs/[model_name]/[params_file_name]  [Mode]

# 功能：运行测试
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/test_train_inference_python.sh configs/[model_name]/[params_file_name]  [Mode]
```

例如，测试基本训练预测功能的`lite_train_infer`模式，运行：
```shell
# 准备数据
bash test_tipc2/prepare.sh ./test_tipc2/configs/fast_rcnn_r50_fpn_1x_coco.txt 'lite_train_infer'
# 运行测试
bash test_tipc2/test_train_inference_python.sh ./test_tipc2/configs/fast_rcnn_r50_fpn_1x_coco.txt 'lite_train_infer'
```

## 4. 开始测试
各功能测试中涉及混合精度等训练相关，及 mkldnn、Tensorrt 等多种预测相关参数配置，请点击下方相应链接了解更多细节和使用教程：  
- [test_train_inference_python 使用](https://github.com/FL77N/RetinaNet-Based-on-PPdet/blob/main/tipc/train_infer_python/README.md) ：测试基于 Python 的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。
- [test_serving 使用](https://github.com/FL77N/RetinaNet-Based-on-PPdet/blob/main/tipc/serving/README.md) ：serving 端的预测功能。
