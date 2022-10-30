'''
<--------------------------------------->
        Created on: 12.01.2022
        @Author: feizzhang
<--------------------------------------->
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ["ATSS"]

@register
class ATSS(BaseArch):
    __category__ = 'architecture'
    __inject__ = ["postprocess", "anchor_generator"]

    def __init__(self,
                 backbone,
                 neck,
                 anchor_generator="AnchorGenerator",
                 head="ATSSHead",
                 postprocess="RetinaNetPostProcess"):
        super(ATSS, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.anchor_generator = anchor_generator
        self.head = head
        self.postprocess = postprocess

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        anchor_generator = create(cfg["anchor_generator"])
        num_anchors = anchor_generator.num_anchors

        kwargs = {"num_anchors": num_anchors}
        head = create(cfg['head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "head": head,
            "anchor_generator": anchor_generator
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(body_feats)
        anchors = self.anchor_generator(fpn_feats)
        cls_pred_list, reg_pred_list, centerness_pred_list = self.head(fpn_feats)

        if not self.training:
            bboxes = self.postprocess(
                cls_pred_list, 
                reg_pred_list, 
                anchors,
                self.inputs
            )

            return bboxes
        else:
            return cls_pred_list, reg_pred_list, centerness_pred_list, anchors

    def get_loss(self):
        cls_pred_list, reg_pred_list, centerness_pred_list, anchors = self._forward()
        
        loss_dict = self.head.get_loss(cls_pred_list, reg_pred_list, centerness_pred_list, anchors, self.inputs)
        total_loss = sum(loss_dict.values())

        loss_dict.update({"loss": total_loss})

        return loss_dict

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}

        return output