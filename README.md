# RetinaNet
   * [RetinaNet](#resnet)
      * [一、简介](#一简介)
      * [二、复现精度及数据集](#二复现精度)
      * [三、环境依赖](#三环境依赖)
      * [四、快速开始](#四快速开始)
         * [step1: 训练](#step2-训练)
         * [step2: 评估](#step3-评估)
         * [step3: 测试](#step4-使用预训练模型预测)
      * [五、代码结构与详细说明](#五代码结构与详细说明)
         * [5.1 代码结构](#51-代码结构)
         * [5.2 参数说明](#52-参数说明)
         * [5.3 训练流程](#53-训练流程)
            * [单机训练](#单机训练)
            * [多机训练](#多机训练)
            * [训练输出](#训练输出)
         * [5.4 评估流程](#54-评估流程)
         * [5.5 测试流程](#55-测试流程)
         * [5.6 使用预训练模型预测](#56-使用预训练模型预测)
         * [5.7 TIPC 测试](#57-TIPC测试)
       * [六、Citations](#六Citations)
       * [七、TODO](#七TODO)

## 一、简介

本项目基于 PPDet 复现 RetinaNet，RetinaNet 是非常经典的单阶段目标检测算法，其使用的 Focal Loss 被众多单阶段网络采用以解决正负样本不平衡的问题。

**论文:**
- [Focal loss for dense object detection](https://arxiv.org/abs/1708.02002)

**参考项目：**
- [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/configs/Base-RetinaNet.yaml)

## 二、复现精度

>训练数据集为 [MS-COCO](https://cocodataset.org/#download) train2017 ，测试数据集为 val2017.

Method|Environment|mAP|Epoch|batch_size|config|Dataset
:--:|:--:|:--:|:--:|:--:|:--:|:--:
r50_fpn_1x_ss_training|Tesla V-100 x 8 <br/>([Facebook official](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md))|35.7|12|16|[retinanet_R-50-FPN_1x.yaml](https://github.com/facebookresearch/Detectron/blob/main/configs/12_2017_baselines/retinanet_R-50-FPN_1x.yaml)|COCO
r50_fpn_1x_ss_training|**Tesla V-100 x 4**|**37.0**|12|16|[-](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md)|COCO
r50_fpn_1x_ms_training|Tesla V-100 x 8 <br/>([Facebook official](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md))|37.4|12|16|[retinanet_R_50_FPN_1x.yaml](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml)|COCO
r50_fpn_1x_ms_training|**Tesla V-100 x 4**|**37.4**|12|16|[retinanet_r50_fpn_1x_coco.yml](https://github.com/FL77N/RetinaNet-Based-on-PPdet/blob/main/configs/retinanet/retinanet_r50_fpn_1x_coco.yml)|COCO

**模型下载**
* The single scale training best model is saved to: [Baidu Aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/104154)
* The multi scale training best model is saved to: [Baidu Aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/104021)

## 三、环境依赖

- 硬件：GPU、CPU

- 框架：
  - PaddlePaddle >= 2.1.2 PaddleDetection >= 2.1.2

## 四、快速开始

### step1: clone 

```bash
# clone this repo
git clone https://github.com/FL77N/RetinaNet-Based-on-PPdet.git
cd RetinaNet-Based-on-PPdet
```
**安装依赖**
```bash
pip install -r ./requirements.txt
```

### step2: 训练
* <font color=pink>single gpu</font> 
```bash
python ./tools/train.py -c .configs/retinanet/retinanet_r50_fpn_1x_coco.yml --eval
```
* <font color=red>mutil gpu</font>
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 ./tools/train.py -c ./configs/retinanet/retinanet_r50_fpn_1x_coco.yml --eval
```

此时的输出为：
```
ppdet.engine INFO: Epoch: [0] [ 0/63] learning_rate: 0.001000 loss_bbox_cls: 4.332087 loss_bbox_reg: 0.630385 loss: 4.962472 eta: 0:00:11 batch_cost: 0.1884 data_cost: 0.0004 ips: 10.6183 images/s
...
```

### step3: 评估
* <font color=pink>eval</font>
```bash
python ./tools/eval.py -c ./configs/retinanet/retinanet_r50_fpn_1x_coco.yml -o weights=best_model.pdparams
```
此时的输出为：
```
ppdet.engine INFO: Eval iter: 0
ppdet.metrics.metrics INFO: The bbox result is saved to bbox.json.
loading annotations into memory...
...
```

### step4: 使用预训练模型预测

configs/your dir/your config.yml 为预测模型的配置文件，通过 infer_img 指定需要预测的图片，通过 weights 加载训练好的模型。

```bash
python tools/infer.py -c configs/your dir/your config.yml --infer_img=your image.jpg -o weights=your best model.pdparams
```

## 五、代码结构与详细说明

### 5.1 代码结构

```
├─config                          # 配置
├─dataset                         # 数据集加载
├─deploy                          # 模型部署
├─output_infer                    # infer 可视化输出
├─ppdet                           # 模型
├─tipc                            # tipc 测试脚本
├─tools                           # 训练、推理、预测
│  README.md                      # readme
│  requirement.txt                # 依赖
│  train-log.log                  # 训练日志
```

### 5.2 参数说明

[训练、推理、测试相关参数设置](https://github.com/FL77N/RetinaNet-Based-on-PPdet/blob/main/configs/retinanet/_base_/retinanet_reader.yml)

[优化器相关参数设置](https://github.com/FL77N/RetinaNet-Based-on-PPdet/blob/main/configs/retinanet/_base_/optimizer_1x.yml)

[数据集加载相关设置](https://github.com/FL77N/RetinaNet-Based-on-PPdet/blob/main/configs/datasets/coco_detection.yml)

### 5.3 训练流程

#### 单机训练
```bash
python ./tools/train.py -c your_config_file
```

#### 多机训练
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3... ./tools/train.py -c your_config_file
```

#### 训练输出
执行训练开始后，将得到类似如下的输出。每一轮`batch`训练将会打印当前epoch、step以及loss值。
```text
ppdet.engine INFO: Epoch: [0] [ 0/63] learning_rate: 0.001000 loss_bbox_cls: 4.332087 loss_bbox_reg: 0.630385 loss: 4.962472 eta: 0:00:11 batch_cost: 0.1884 data_cost: 0.0004 ips: 10.6183 images/s
...
```

### 5.4 评估流程

```bash
python ./tools/eval.py -c your_config_file -o weights=your_best_model.pdparams
```

此时的输出为：
```
ppdet.engine INFO: Eval iter: 0
ppdet.metrics.metrics INFO: The bbox result is saved to bbox.json.
loading annotations into memory...
```

### 5.5 测试流程

```bash
python tools/infer.py -c configs/your dir/your config.yml --infer_img=your image.jpg -o weights=your best model.pdparams
```
此时的输出结果保存在`output`下面


### 5.6 使用预训练模型预测

使用预训练模型预测的流程如下：

**step1:** 下载预训练模型
* The multi scale training best model are saved to: [Baidu Aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/104021)

**step2:** 使用预训练模型完成预测
```bash
python tools/infer.py -c ./configs/retinanet/retinanet_r50_fpn_1x_coco.yml --infer_img=your image.jpg -o weights=best_model.pdparams
```
### 5.7 TIPC测试

The tutorials of TIPC：[Python](https://github.com/FL77N/RetinaNet-Based-on-PPdet/blob/main/tipc/train_infer_python)、[Serving](https://github.com/FL77N/RetinaNet-Based-on-PPdet/tree/main/tipc/serving)

The results of TIPC: [Python](https://github.com/FL77N/RetinaNet-Based-on-PPdet/blob/main/tipc/train_infer_python/output)、[Serving](https://github.com/FL77N/RetinaNet-Based-on-PPdet/tree/main/tipc/serving/output)

[More detail about PPDet](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/docs/tutorials)

## 六、Citations
```
@misc{linICCV17retinanet,
      Author = {Tsung-Yi Lin},
      Title = {RetinaNet},
      Booktitle = {International Conference on Computer Vision ({ICCV})},
      Year = {2017}
}
```
## 七、TODO
* C++ 部署
