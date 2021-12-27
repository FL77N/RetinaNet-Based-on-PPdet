# Linux端基础训练预测功能测试

Linux端基础训练预测功能测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。


## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 | 多机多卡 | 模型压缩（单机多卡） |
|  :----  |   :----  |    :----  |  :----   |  :----   |  :----   |
|  Retinanet  | retinanet_r50_fpn_1x_coco | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | - |
|  Retinanet  | retinanet_mobilenet_v1_fpn_1x_coco | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | - |
|  Retinanet  | retinanet_mobilenet_v3_small_fpn_1x_coco | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | - |
|  Retinanet  | retinanet_mobilenet_v3_large_fpn_1x_coco | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | - |
|  Retinanet  | retinanet_shufflenet_v2_fpn_1x_coco | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | - |


- 预测相关：

| 模型类型 |device | batchsize | tensorrt | mkldnn | cpu多线程 |
|  ----   |  ---- |   ----   |  :----:  |   :----:   |  :----:  |
| 正常模型 | GPU | 1 | - | - | - |
| 正常模型 | CPU | 1 | - | fp32/fp16 | 支持 |

## 2. 测试流程

### 2.1 安装依赖
- 安装PaddlePaddle >= 2.2
- 安装PaddleDetection依赖
    ```
    pip install -r ./requirements.txt
    pip install -r ./tipc/requirements.txt
    ```
- 安装autolog（规范化日志输出工具）
    ```
    git clone https://github.com/LDOUBLEV/AutoLog
    cd AutoLog
    pip install -r ./requirements.txt
    python setup.py bdist_wheel
    pip install ./dist/auto_log-1.0.0-py3-none-any.whl
    ```


### 2.2 功能测试
先运行`prepare.sh`准备数据和模型，然后运行`test_train_inference_python.sh`进行测试，最终在```test_tipc/output```目录下生成`python_infer_*.log`格式的日志文件，
以 fast_rcnn_r50_fpn_1x_coco 为例。

`test_train_inference_python.sh`包含4种运行模式，每种模式的运行数据不同，分别用于测试速度和精度，分别是：

- 模式1：lite_train_lite_infer，使用少量数据训练，用于快速验证训练到预测的走通流程，不验证精度和速度；
```shell
bash tipc/train_infer_python/prepare.sh tipc/train_infer_python/configs/retinanet/retinanet_r50_fpn_1x_coco.txt 'lite_train_lite_infer'
bash tipc/train_infer_python/test_train_inference_python.sh tipc/train_infer_python/configs/retinanet/retinanet_r50_fpn_1x_coco.txt 'lite_train_lite_infer'
```

- 模式2：lite_train_whole_infer，使用少量数据训练，一定量数据预测，用于验证训练后的模型执行预测，预测速度是否合理；lite_train_whole_infer
```shell
bash tipc/train_infer_python/prepare.sh tipc/train_infer_python/configs/retinanet/retinanet_r50_fpn_1x_coco.txt 'lite_train_whole_infer'
bash tipc/train_infer_python/test_train_inference_python.sh tipc/train_infer_python/configs/retinanet/retinanet_r50_fpn_1x_coco.txt 'lite_train_whole_infer'
```

- 模式3：whole_train_whole_infer，CE： 全量数据训练，全量数据预测，验证模型训练精度，预测精度，预测速度；
```shell
bash tipc/train_infer_python/prepare.sh tipc/train_infer_python/configs/retinanet/retinanet_r50_fpn_1x_coco.txt 'whole_train_whole_infer'
bash tipc/train_infer_python/test_train_inference_python.sh tipc/train_infer_python/configs/retinanet/retinanet_r50_fpn_1x_coco.txt 'whole_train_whole_infer'
```

- 模式4：whole_infer，不训练，全量数据预测，走通开源模型评估、动转静，检查inference model预测时间和精度;
```shell
bash tipc/train_infer_python/prepare.sh tipc/train_infer_python/configs/retinanet/retinanet_r50_fpn_1x_coco.txt 'whole_infer'
bash tipc/train_infer_python/test_train_inference_python.sh tipc/train_infer_python/configs/retinanet/retinanet_r50_fpn_1x_coco.txt 'whole_infer'
```

运行相应指令后，在`tipc/train_infer_python/output`文件夹下自动会保存运行日志。如'lite_train_lite_infer'模式下，会运行训练+推理的链条，因此，在`tipc/train_infer_python/output`文件夹有以下文件：
```
tipc/train_infer_python/output/retinanet_r50/
|- results_python.log    # 运行指令状态的日志
|- norm_train_gpus_0_autocast_null/  # GPU 0号卡上正常训练的训练日志和模型保存文件夹
|- pact_train_gpus_0_autocast_null/  # GPU 0号卡上量化训练的训练日志和模型保存文件夹
......
|- python_infer_cpu_usemkldnn_True_threads_1_precision_fluid_batchsize_1.log  # CPU 上开启 Mkldnn 线程数设置为 1，测试 batch_size=1 条件下的预测运行日志
|- python_infer_gpu_precision_trt_fp16_batchsize_1.log # GPU上开启TensorRT，测试 batch_size=1 的半精度预测日志
......
```

其中`results_python.log`中包含了每条指令的运行状态，如果运行成功会输出：
```
Run successfully with command - python3.7 tools/train.py -c configs/retinanet/retinanet_r50_fpn_1x_coco.yml -o use_gpu=True save_dir=tipc/train_infer_python/output/retinanet_r50/norm_train_gpus_0_autocast_null epoch=2 pretrain_weights=weights/best_model.pdparams TrainReader.batch_size=2 filename=retinanet_r50_fpn_1x_coco  !
Run successfully with command - python3.7 tools/eval.py -c configs/retinanet/retinanet_r50_fpn_1x_coco.yml -o weights=tipc/train_infer_python/output/retinanet_r50/norm_train_gpus_0_autocast_null/retinanet_r50_fpn_1x_coco/model_final.pdparams use_gpu=True  !
......
```
如果运行失败，会输出：
```
Run failed with command - python3.7 tools/train.py -c configs/retinanet/retinanet_r50_fpn_1x_coco.yml -o use_gpu=True save_dir=tipc/train_infer_python/output/retinanet_r50/norm_train_gpus_0_autocast_null epoch=2 pretrain_weights=weights/best_model.pdparams TrainReader.batch_size=2 filename=retinanet_r50_fpn_1x_coco  !
Run failed with command - python3.7 tools/eval.py -c configs/retinanet/retinanet_r50_fpn_1x_coco.yml -o weights=tipc/train_infer_python/output/retinanet_r50/norm_train_gpus_0_autocast_null/retinanet_r50_fpn_1x_coco/model_final.pdparams use_gpu=True  !
......
```
可以很方便的根据`results_python.log`中的内容判定哪一个指令运行错误。

最后可得到预测图片的 infer 打印结果如下：
```
Found 2 inference images in total.
im_bboxes_num 100
class_id:0, confidence:0.9472, left_top:[280.60,5.54],right_bottom:[637.31,395.06]
class_id:27, confidence:0.8061, left_top:[413.43,171.45],right_bottom:[474.45,294.88]
class_id:39, confidence:0.6352, left_top:[539.17,4.76],right_bottom:[558.70,65.39]
class_id:39, confidence:0.6168, left_top:[520.19,3.65],right_bottom:[539.09,67.33]
class_id:39, confidence:0.5056, left_top:[571.42,0.07],right_bottom:[583.28,41.20]
class_id:48, confidence:0.6862, left_top:[203.21,202.27],right_bottom:[369.51,320.14]
class_id:48, confidence:0.6061, left_top:[17.39,249.37],right_bottom:[212.06,329.80]
class_id:48, confidence:0.5586, left_top:[219.65,199.61],right_bottom:[359.93,269.17]
save result to: ./output_infer/python/retinanet_r50/000000575930.jpg
im_bboxes_num 100
class_id:0, confidence:0.8881, left_top:[237.50,419.11],right_bottom:[307.30,611.46]
class_id:33, confidence:0.9035, left_top:[144.65,78.06],right_bottom:[201.04,148.82]
save result to: ./output_infer/python/retinanet_r50/000000564177.jpg
```
其中 infer 的结果会保存至 output_infer/python/retinanet_r50/

## 3. 更多教程
本文档为功能测试用，更丰富的训练预测使用教程请参考：  
[模型训练](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/docs/tutorials)  
[预测部署](../../deploy/README.md)
