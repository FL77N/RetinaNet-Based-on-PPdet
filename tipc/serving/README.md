# PaddleServing预测功能测试

PaddleServing预测功能测试的主程序为`test_serving.sh`，可以测试基于PaddleServing的部署功能。

## 1. 测试结论汇总

基于训练是否使用量化，进行本测试的模型可以分为`正常模型`和`量化模型`，这两类模型对应的Serving预测功能汇总如下：

| 模型类型 |device | batchsize | tensorrt | mkldnn | cpu多线程 |
|  ----   |  ---- |   :----:   |  :----:  |   :----:   |  :----:  |
| 正常模型 | GPU | 1 | - | - | - |

## 2. 测试流程

### 2.1 准备环境

* 首先准备docker环境，AIStudio环境已经安装了合适的docker。如果是非AIStudio环境，请[参考文档](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/environment.md)中的 "1.3.2 Docker环境配置" 安装docker环境。

* 然后安装Paddle Serving三个安装包，paddle-serving-server，paddle-serving-client 和 paddle-serving-app。

```bash
wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_server_gpu-0.7.0.post102-py3-none-any.whl
# 如果是 cuda 10.1 
# wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_server_gpu-0.7.0.post101-py3-none-any.whl
pip install paddle_serving_server_gpu-0.7.0.post102-py3-none-any.whl

wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_client-0.7.0-cp37-none-any.whl
pip install paddle_serving_client-0.7.0-cp37-none-any.whl

wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_app-0.7.0-py3-none-any.whl
pip install paddle_serving_app-0.7.0-py3-none-any.whl
```

如果希望获取Paddle Serving Server更多不同运行环境的whl包下载地址，请参考：[下载页面](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Latest_Packages_CN.md)


### 2.2 准备模型

使用训推一体化进行动转静，得到静态图模型。也可以使用 export_model.py 导出模型，命令见[PPDet 第八点](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md) ，导出的模型须于[配置文件第五行](https://github.com/FL77N/RetinaNet-Based-on-PPdet/blob/main/tipc/serving/configs/retinanet_r50_fpn_1x_coco.txt)对应。

```bash
bash tipc/train_infer_python/prepare.sh tipc/train_infer_python/configs/retinanet/retinanet_r50_fpn_1x_coco.txt 'whole_infer'
bash tipc/train_infer_python/test_train_inference_python.sh tipc/train_infer_python/configs/retinanet/retinanet_r50_fpn_1x_coco.txt 'whole_infer'
```

### 2.3 测试功能

运行`test_serving.sh`进行测试，最终在`tipc/output`目录下生成`serving_infer_*.log`后缀的日志文件。

```bash
bash tipc/test_serving.sh tipc/serving/configs/retinanet_r50_fpn_1x_coco.txt
```  

#### 运行结果

各测试的运行情况会打印在 `tipc/output/results_serving.log` 中：
运行成功时会输出：

```
Run successfully with command - python3.7 pipeline_http_client.py --img_path=../../dataset/coco/test2017/000000575930.jpg> ../../tipc/serving/output/server_infer_gpu_pipeline_http_usetrt_null_precision_null_batchsize_1.log 2>&1!
Run successfully  with command - xxxxx
...
```

运行失败时会输出：

```
Run failed with command - python3.7 pipeline_http_client.py --image_dir=../../doc/imgs > ../../tests/output/server_infer_cpu_usemkldnn_True_threads_1_batchsize_1.log 2>&1 !
Run failed with command - python3.7 pipeline_http_client.py --image_dir=../../doc/imgs > ../../tests/output/server_infer_cpu_usemkldnn_True_threads_6_batchsize_1.log 2>&1 !
Run failed with command - xxxxx
...
```

详细的预测结果会存在 tipc/output/ 文件夹下，例如`server_infer_gpu_pipeline_http_usetrt_null_precision_null_batchsize_1.log`中会返回类别ID、置信度、预测框左上坐标及右下坐标:

```
{'err_no': 0, 'err_msg': '', 'key': ['class_id', 'confidence', 'left_top', 'right_bottom'], 
'value': [
'[0, 27, 39, 39, 39, 48, 48, 48]', 
'[0.94724214, 0.80608857, 0.6352301, 0.6167962, 0.50561297, 0.68617314, 0.6061374, 0.55860823]', 
'[[280.60333, 5.541393], [413.42865, 171.45193], [539.1742, 4.7642527], [520.187, 3.6457572], [571.4183, 0.07210945], [203.21043, 202.2723], [17.387094, 249.37366], [219.64606, 199.61432]]', 
'[[637.30597, 395.05698], [474.4487, 294.87994], [558.69916, 65.38775], [539.08765, 67.33221], [583.2792, 41.201984], [369.5113, 320.1367], [212.05835, 329.80197], [359.92694, 269.16583]]'
], 
'tensors': []}
```

可于 python 端预测对比，完全一致则预测正确。

```
class_id:0, confidence:0.9472, left_top:[280.60,5.54],right_bottom:[637.31,395.06]
class_id:27, confidence:0.8061, left_top:[413.43,171.45],right_bottom:[474.45,294.88]
class_id:39, confidence:0.6352, left_top:[539.17,4.76],right_bottom:[558.70,65.39]
class_id:39, confidence:0.6168, left_top:[520.19,3.65],right_bottom:[539.09,67.33]
class_id:39, confidence:0.5056, left_top:[571.42,0.07],right_bottom:[583.28,41.20]
class_id:48, confidence:0.6862, left_top:[203.21,202.27],right_bottom:[369.51,320.14]
class_id:48, confidence:0.6061, left_top:[17.39,249.37],right_bottom:[212.06,329.80]
class_id:48, confidence:0.5586, left_top:[219.65,199.61],right_bottom:[359.93,269.17]
```
## 3. 更多教程

本文档为功能测试用，更详细的 Serving 预测使用请参考原 repo 教程：[AlexNet 服务化部署](https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step5/AlexNet_paddle/README.md)  

