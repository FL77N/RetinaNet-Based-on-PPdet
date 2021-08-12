## Introduction
A single stage object detection model is implemented on paddle 2.1.0 and paddledetection 2.1.0

## Results
Method|Environment|mAP|Epoch|Dataset
:--:|:--:|:--:|:--:|:--:
R50_fpn_1x|Tesla V-100 x 8 ([Facebook official](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md))|37.4|12|COCO
R50_fpn_1x|**Tesla V-100 x 4**|<font color=red>37.4</font>|12|COCO

## Model and Pretrain Model
* The best model is saved to: [Baidu Aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/104021)
* Torch and paddle [pretrain model](https://aistudio.baidu.com/aistudio/datasetdetail/103882)

## Train
* <font color=pink>single gpu</font> 
    
    ```python PaddleDetection/tools/train.py -c PaddleDetection/configs/retinanet/retinanet_r50_fpn_1x_coco.yml --eval```
* <font color=red>mutil gpu</font>
   
   ```python -m paddle.distributed.launch --gpus 0,1,2,3 PaddleDetection/tools/train.py -c PaddleDetection/configs/retinanet/retinanet_r50_fpn_1x_coco.yml --eval```

## Eval

```python tools/eval.py -c configs/sparse_rcnn/sparse_rcnn_r50_fpn_3x_pro100_coco.yml```
