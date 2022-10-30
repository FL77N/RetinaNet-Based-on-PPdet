'''
<--------------------------------------->
        Created on: 03.11.2021
        @Author: feizzhang
<--------------------------------------->
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn
from paddle.nn.initializer import Normal, Constant
from paddle import ParamAttr

from ppdet.core.workspace import register


class Scale(nn.Layer):
    def __init__(self, init_value):
        super(Scale, self).__init__()
        self.scale = self.create_parameter(
            shape=[1],
            attr=ParamAttr(initializer=Constant(value=init_value)),
            dtype="float32"
        )

    def forward(self, inputs):
        out = inputs * self.scale
        return out


@register
class ATSSHead(nn.Layer):
    __inject__ = ['loss_func']
    def __init__(
        self, 
        in_channels,
        num_classes,
        num_convs,
        prior_prob,
        loss_func,
        num_anchors=1,
    ):
        super(ATSSHead, self).__init__()
        self.loss_func = loss_func
        self.num_classes = num_classes

        cls_convs = []
        reg_convs = []
        for i in range(num_convs):
            cls_convs.append(
                nn.Conv2D(
                   in_channels=in_channels, 
                   out_channels=in_channels, 
                   kernel_size=3, 
                   stride=1, 
                   padding=1,
                   weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)),
                   bias_attr=paddle.ParamAttr(initializer=Constant(0.))
                )
            )

            cls_convs.append(nn.GroupNorm(32, in_channels))
            cls_convs.append(nn.ReLU())

            reg_convs.append(
                nn.Conv2D(
                   in_channels=in_channels, 
                   out_channels=in_channels, 
                   kernel_size=3, 
                   stride=1, 
                   padding=1,
                   weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)),
                   bias_attr=paddle.ParamAttr(initializer=Constant(0.))
                )
            )

            reg_convs.append(nn.GroupNorm(32, in_channels))
            reg_convs.append(nn.ReLU())

        self.add_sublayer("cls_convs", nn.Sequential(*cls_convs))
        self.add_sublayer("reg_convs", nn.Sequential(*reg_convs))

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.cls_score = nn.Conv2D(
            in_channels=in_channels, 
            out_channels=num_anchors*self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)),
            bias_attr=paddle.ParamAttr(initializer=Constant(bias_value))
        )
        self.bbox_pred = nn.Conv2D(
            in_channels=in_channels, 
            out_channels=num_anchors*4,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)),
            bias_attr=paddle.ParamAttr(initializer=Constant(0.))
        )
        self.centerness = nn.Conv2D(
            in_channels=in_channels, 
            out_channels=num_anchors*1,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)),
            bias_attr=paddle.ParamAttr(initializer=Constant(0.))
        )

        self.scales = nn.LayerList([Scale(init_value=1.0) for _ in range(5)])

    @classmethod
    def from_config(cls, cfg, num_anchors):
        return {'num_anchors': num_anchors}
    
    def get_loss(self, cls_pred_list, reg_pred_list, centerness_pred_list, anchors, inputs):
        return self.loss_func(cls_pred_list, reg_pred_list, centerness_pred_list, anchors, inputs)

    def forward(self, feats):
        cls_pred_list = []
        reg_pred_list = []
        centerness_pred_list = []

        for i, feat in enumerate(feats):
            cls_feat = self.cls_convs(feat)
            reg_feat = self.reg_convs(feat)

            cls_per_lvl = self.cls_score(cls_feat)
            reg_per_lvl = self.scales[i](self.bbox_pred(reg_feat))
            centerness_per_lvl = self.centerness(reg_feat)

            B, _, H, W = cls_per_lvl.shape
            cls_per_lvl = cls_per_lvl.transpose([0, 2, 3, 1]).reshape([B, -1, self.num_classes])
            reg_per_lvl = reg_per_lvl.transpose([0, 2, 3, 1]).reshape([B, -1, 4])
            centerness_per_lvl = centerness_per_lvl.transpose([0, 2, 3, 1]).reshape([B, -1, 1])
            cls_pred_list.append(cls_per_lvl)
            reg_pred_list.append(reg_per_lvl)
            centerness_pred_list.append(centerness_per_lvl)
        
        return cls_pred_list, reg_pred_list, centerness_pred_list