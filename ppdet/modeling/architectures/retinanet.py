'''
    Created on: 05.08.2021
    @Author: feizzhang
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ["RetinaNet"]


@register
class RetinaNet(BaseArch):
    __category__ = 'architecture'
    __inject__ = ["postprocess", "anchor_generator"]

    def __init__(self,
                 backbone,
                 neck,
                 anchor_generator="AnchorGenerator",
                 head="RetinaNetHead",
                 postprocess="RetinaNetPostProcess"):
        super(RetinaNet, self).__init__()
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

        kwargs = {'input_shape': neck.out_shape[1:], "num_anchors": num_anchors}  # ppdet bug: get None
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
        pred_scores, pred_boxes = self.head(fpn_feats)

        if not self.training:
            bboxes = self.postprocess(
                pred_scores, 
                pred_boxes, 
                anchors,
                self.inputs
            )

            return bboxes
        else:
            return anchors, pred_scores, pred_boxes

    def get_loss(self):
        anchors, pred_scores, pred_boxes = self._forward()
        
        loss_dict = self.head.losses(anchors, [pred_scores, pred_boxes], self.inputs)
        total_loss = sum(loss_dict.values())

        loss_dict.update({"loss": total_loss})

        return loss_dict

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}

        return output


