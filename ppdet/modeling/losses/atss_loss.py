'''
<--------------------------------------->
        Created on: 12.01.2022
        @Author: feizzhang
<--------------------------------------->
'''

import os
import math
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ppdet.modeling.bbox_utils import boxes_ious, delta2bbox

__all__ = ["ATSSLoss"]

# <----------------------- paddle bug -----------------------> #
'''
[!] paddle.gather 即便输入的小标超出了 tensor 的边界也会去 gather。
但 gather 了一些奇怪的值
[!] paddle.tile 会使显存不断增长变得很大。
'''
# <----------------------------------------------------------> #

INF = 100000000

@register
class ATSSLoss(nn.Layer):
    def __init__(
        self,
        aspect_ratios,
        anchor_size,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2,
        topk=9, 
        num_classes=80,
        reg_loss_weight=2,
        bbox_reg_weights=(10, 10, 5, 5)   
    ):
        super(ATSSLoss, self).__init__()
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.aspect_ratios = aspect_ratios
        self.anchor_size = anchor_size
        self.topk = topk
        self.num_classes = num_classes
        self.bbox_reg_weights = bbox_reg_weights
        self.reg_loss_weight = 2.0

    def boxes_ious(self, boxes1, boxes2):
        assert boxes1.shape[-1] == 4
        assert boxes2.shape[-1] == 4

        boxes1_wh = (boxes1[..., 2:] - boxes1[..., :2]).clip(min=0)
        boxes1_area = boxes1_wh.prod(axis=-1)
        boxes2_wh = (boxes2[..., 2:] - boxes2[..., :2]).clip(min=0)
        boxes2_area = boxes2_wh.prod(axis=-1)

        lt = np.maximum(boxes1[:, None, :2], boxes2[..., :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[..., 2:])

        inter_wh = (rb - lt).clip(min=0)
        inter_area = inter_wh.prod(axis=-1)

        union_area = boxes1_area[..., None] + boxes2_area - inter_area + 1e-6

        ious = (inter_area / union_area).clip(min=0)

        return ious

    def prepare_targets_np(self, gt, anchors):
        batch_gt_box = gt["gt_bbox"]
        batch_gt_class = gt["gt_class"]

        num_anchors_per_loc = len(self.anchor_size[0]) * len(self.aspect_ratios)
        num_anchors_per_lvl = [len(anchors_per_lvl) for anchors_per_lvl in anchors]
        anchors_per_img = paddle.concat(anchors).numpy()
        anchors_cx_per_img = (anchors_per_img[:, 2] + anchors_per_img[:, 0]) / 2.0
        anchors_cy_per_img = (anchors_per_img[:, 3] + anchors_per_img[:, 1]) / 2.0
        anchor_points = np.stack((anchors_cx_per_img, anchors_cy_per_img), axis=1)

        cls_tgt_list = []
        reg_tgt_list = []
        anchor_tgt_list = []
        for gt_box_per_img, gt_class_per_img in zip(batch_gt_box, batch_gt_class):
            gt_box_per_img = gt_box_per_img.numpy()
            gt_class_per_img = gt_class_per_img.flatten().numpy()

            num_gt = gt_box_per_img.shape[0]

            ious = self.boxes_ious(anchors_per_img, gt_box_per_img)

            gt_cx = (gt_box_per_img[:, 2] + gt_box_per_img[:, 0]) / 2.0
            gt_cy = (gt_box_per_img[:, 3] + gt_box_per_img[:, 1]) / 2.0
            gt_points = np.stack((gt_cx, gt_cy), axis=1)

            distances = np.sqrt(
                np.power(anchor_points[:, None, :] - gt_points[None, :, :], 2).sum(-1)
            )

            candidate_idxs = []
            start_idx = 0
            for lvl in range(len(num_anchors_per_lvl)):
                end_idx = start_idx + num_anchors_per_lvl[lvl]
                distances_per_lvl = distances[start_idx:end_idx, :]
                topk = min(self.topk * num_anchors_per_loc, num_anchors_per_lvl[lvl])
                _, topk_idxs_per_lvl = _topk(distances_per_lvl, topk, axis=0, largest=False)
                candidate_idxs.append(topk_idxs_per_lvl + start_idx)
                start_idx = end_idx
            candidate_idxs = np.concatenate(candidate_idxs, axis=0)

            candidate_ious = ious[candidate_idxs, np.arange(num_gt)]
            iou_mean_per_gt = candidate_ious.mean(0)
            iou_std_per_gt = candidate_ious.std(0)
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt

            is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

            num_anchors = anchors_per_img.shape[0]
            for gt_idx in range(num_gt):
                candidate_idxs[:, gt_idx] += gt_idx * num_anchors
            e_anchors_cx  = np.broadcast_to(
                anchors_cx_per_img.reshape(1, -1), [num_gt, num_anchors]).reshape(-1)
            e_anchors_cy  = np.broadcast_to(
                anchors_cy_per_img.reshape(1, -1), [num_gt, num_anchors]).reshape(-1)
            candidate_idxs = candidate_idxs.reshape(-1)

            l = e_anchors_cx[candidate_idxs].reshape(-1, num_gt) - gt_box_per_img[:, 0]
            t = e_anchors_cy[candidate_idxs].reshape(-1, num_gt) - gt_box_per_img[:, 1]
            r = gt_box_per_img[:, 2] - e_anchors_cx[candidate_idxs].reshape(-1, num_gt)
            b = gt_box_per_img[:, 3] - e_anchors_cy[candidate_idxs].reshape(-1, num_gt)
            is_in_gts = np.stack([l, t, r, b], axis=1).min(axis=1) > 0.01
            is_pos = is_pos & is_in_gts

            # if an anchor box is assigned to multiple gts,
            # the one with the highest IoU will be selected.
            ious_inf = -INF * np.ones_like(ious).T.reshape(-1)
            index = candidate_idxs.reshape(-1)[is_pos.reshape(-1)]
            ious_inf[index] = ious.T.reshape(-1)[index]
            ious_inf = ious_inf.reshape(num_gt, -1).T

            anchors_to_gt_values = ious_inf.max(axis=1)
            anchors_to_gt_indexs = ious_inf.argmax(axis=1)

            cls_labels_per_img = gt_class_per_img[anchors_to_gt_indexs]
            cls_labels_per_img[anchors_to_gt_values == -INF] = self.num_classes

            reg_tgt_per_img = gt_box_per_img[anchors_to_gt_indexs]

            cls_tgt_list.append(paddle.to_tensor(cls_labels_per_img))
            reg_tgt_list.append(paddle.to_tensor(reg_tgt_per_img))
            anchor_tgt_list.append(paddle.to_tensor(anchors_per_img))

        return cls_tgt_list, reg_tgt_list, anchor_tgt_list

    @paddle.no_grad()
    def prepare_targets(self, gt, anchors):
        batch_gt_box = gt["gt_bbox"]
        batch_gt_class = gt["gt_class"]

        cls_tgt_list = []
        reg_tgt_list = []
        anchors_tgt_list = []

        num_anchors_per_loc = len(self.anchor_size[0]) * len(self.aspect_ratios)
        num_anchors_per_lvl = [len(anchors_per_lvl) for anchors_per_lvl in anchors]

        anchors_per_img = paddle.concat(anchors)
        anchors_cx_per_img = (anchors_per_img[:, 2] + anchors_per_img[:, 0]) / 2.0
        anchors_cy_per_img = (anchors_per_img[:, 3] + anchors_per_img[:, 1]) / 2.0
        anchor_points = paddle.stack((anchors_cx_per_img, anchors_cy_per_img), axis=1)

        for gt_box_per_img, gt_class_per_img in zip(batch_gt_box, batch_gt_class):
            gt_class_per_img = gt_class_per_img.flatten()
            num_gt = gt_box_per_img.shape[0]

            ious, _ = boxes_ious(anchors_per_img, gt_box_per_img, mode="a", type="iou")

            gt_cx = (gt_box_per_img[:, 2] + gt_box_per_img[:, 0]) / 2.0
            gt_cy = (gt_box_per_img[:, 3] + gt_box_per_img[:, 1]) / 2.0
            gt_points = paddle.stack((gt_cx, gt_cy), axis=1)

            distances = (anchor_points.unsqueeze(-2) - gt_points.unsqueeze(0)).pow(2).sum(-1).sqrt()
            # Selecting candidates based on the center distance between anchor box and object
            candidate_idxs = []
            start_idx = 0
            for lvl, anchors_per_lvl in enumerate(anchors):
                end_idx = start_idx + num_anchors_per_lvl[lvl]
                distances_per_lvl = distances[start_idx:end_idx, :]
                topk = min(self.topk * num_anchors_per_loc, num_anchors_per_lvl[lvl])
                _, topk_idxs_per_lvl = paddle.topk(distances_per_lvl, topk, axis=0, largest=False)
                candidate_idxs.append(topk_idxs_per_lvl + start_idx)
                start_idx = end_idx
            candidate_idxs = paddle.concat(candidate_idxs, axis=0)
            # for i in [322, 5253, 324, 5413, 242, 5254, 244, 5414, 402, 5252]:
            #     print(candidate_idxs[i])
            # print(candidate_idxs)
            # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
            candidate_ious = paddle.index_sample(ious.transpose([1, 0]), candidate_idxs.transpose([1, 0])).transpose([1, 0])

            iou_mean_per_gt = candidate_ious.mean(0)
            iou_std_per_gt = candidate_ious.std(0)
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
            is_pos = candidate_ious >= iou_thresh_per_gt.unsqueeze(0)

            # Limiting the final positive samples’ center to object
            
            # e_anchors_cx = paddle.reshape(anchors_cx_per_img, [1, -1])
            # e_anchors_cx = paddle.tile(e_anchors_cx, [num_gt, anchor_num])
            # e_anchors_cx = paddle.reshape(e_anchors_cx, [-1])
            # e_anchors_cy = paddle.reshape(anchors_cy_per_img, [1, -1])
            # e_anchors_cy = paddle.tile(e_anchors_cy, [num_gt, anchor_num])
            # e_anchors_cy = paddle.reshape(e_anchors_cy, [-1])

            e_anchors_cx_c = paddle.gather(anchors_cx_per_img, candidate_idxs.reshape([-1])).reshape([-1, num_gt])
            e_anchors_cy_c = paddle.gather(anchors_cy_per_img, candidate_idxs.reshape([-1])).reshape([-1, num_gt])

            l = e_anchors_cx_c - gt_box_per_img[:, 0]
            t = e_anchors_cy_c - gt_box_per_img[:, 1]
            r = gt_box_per_img[:, 2] - e_anchors_cx_c
            b = gt_box_per_img[:, 3] - e_anchors_cy_c
            # [n_a, n_g] -> [n_a, 4, n_g] -> [n_a, n_g]
            is_in_gts = paddle.stack([l, t, r, b], axis=1).min(axis=1) > 0.01
            # print(is_in_gts)
            is_pos = paddle.logical_and(is_pos, is_in_gts).reshape([-1])
            # print(candidate_idxs)
            anchor_num = anchors_per_img.shape[0]
            for ng in range(num_gt):
                candidate_idxs[:, ng] += ng * anchor_num
            candidate_idxs = candidate_idxs.reshape([-1])
            # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
            # [n_g, n_a]
            ious_inf = paddle.full_like(ious, -INF).transpose([1, 0]).reshape([-1])
            if is_pos.sum() != 0:
                index = paddle.masked_select(candidate_idxs, is_pos)
                ious_inf = paddle.scatter(ious_inf, index, paddle.gather(ious.transpose([1, 0]).reshape([-1]), index))
            # print(index)
            ious_inf = paddle.reshape(ious_inf, [num_gt, -1])
            ious_inf = paddle.transpose(ious_inf, [1, 0])

            # [n_a]
            anchors_to_gt_values = ious_inf.max(1)
            anchors_to_gt_indexs = paddle.argmax(ious_inf, axis=1)
            cls_labels_per_img = paddle.gather(gt_class_per_img, anchors_to_gt_indexs)
            cls_labels_per_img = paddle.where(
                anchors_to_gt_values == -INF,
                paddle.full_like(cls_labels_per_img, self.num_classes),
                cls_labels_per_img
            )

            reg_tgt_per_img = paddle.gather(gt_box_per_img, anchors_to_gt_indexs)
            # print(paddle.masked_select(reg_tgt_per_img, (cls_labels_per_img>0).unsqueeze(-1).tile([1,4])).reshape([-1,4]))
            cls_tgt_list.append(cls_labels_per_img)
            reg_tgt_list.append(reg_tgt_per_img)
            anchors_tgt_list.append(anchors_per_img)

        return cls_tgt_list, reg_tgt_list, anchors_tgt_list

    def compute_centerness_targets(self, gts, anchors):
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l = anchors_cx - gts[:, 0]
        t = anchors_cy - gts[:, 1]
        r = gts[:, 2] - anchors_cx
        b = gts[:, 3] - anchors_cy
        left_right = paddle.stack([l, r], axis=1)
        top_bottom = paddle.stack([t, b], axis=1)
        centerness = paddle.sqrt(
            (left_right.min(axis=-1) / left_right.max(axis=-1)) * (top_bottom.min(axis=-1) / top_bottom.max(axis=-1))
        )
        assert not paddle.isnan(centerness).any()
        return centerness

    def forward(self, cls_pred_list, reg_pred_list, centerness_pred_list, anchors, inputs):
        cls_tgt_list, reg_tgt_list, anchor_tgt_list = self.prepare_targets(inputs, anchors)
        N = cls_tgt_list[0].shape[0]
        cls_pred = paddle.concat(cls_pred_list, axis=1).reshape([-1, self.num_classes])
        delta_pred = paddle.concat(reg_pred_list, axis=1).reshape([-1, 4])
        centerness_pred = paddle.concat(centerness_pred_list, axis=1).reshape([-1])

        cls_tgt_flatten = paddle.concat(cls_tgt_list)
        reg_tgt_flatten = paddle.concat(reg_tgt_list)
        anchors_flatten = paddle.concat(anchor_tgt_list)

        pos_mask = paddle.logical_and(cls_tgt_flatten < self.num_classes, cls_tgt_flatten >= 0)
        pos_idx = paddle.nonzero(pos_mask).flatten()

        num_pos = max(pos_idx.shape[0], 1.0)
        cls_tgt = F.one_hot(cls_tgt_flatten, num_classes=self.num_classes + 1)[:, :-1]
        cls_tgt.stop_gradient = True

        cls_loss = F.sigmoid_focal_loss(cls_pred,
                                        cls_tgt,
                                        alpha=self.focal_loss_alpha,
                                        gamma=self.focal_loss_gamma,
                                        reduction='sum') / num_pos

        delta_pred = paddle.gather(delta_pred, pos_idx)
        reg_tgt_flatten = paddle.gather(reg_tgt_flatten, pos_idx)
        anchors_flatten = paddle.gather(anchors_flatten, pos_idx)
        centerness_pred = paddle.gather(centerness_pred, pos_idx)
        centerness_targets = self.compute_centerness_targets(reg_tgt_flatten, anchors_flatten)
        sum_centerness_targets = centerness_targets.sum()

        anchors_flatten.stop_gradient = True
        reg_tgt_flatten.stop_gradient = True
        centerness_targets.stop_gradient = True

        if pos_idx.numel() > 0:
            reg_pred = delta2bbox(delta_pred, anchors_flatten, self.bbox_reg_weights).reshape([-1, 4])
            reg_loss = ((1 - boxes_ious(reg_pred, reg_tgt_flatten, mode="b", type="giou")[0]) * centerness_targets).sum() / sum_centerness_targets
            centerness_loss = F.binary_cross_entropy_with_logits(centerness_pred, centerness_targets, reduction="sum") / num_pos
        else:
            reg_loss = reg_pred.sum()
            centerness_loss = centerness_pred.sum()

        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss * self.reg_loss_weight,
            "centerness_loss": centerness_loss
        }


def _topk(input, k, axis=1, largest=True):
    x = -input if largest else input
    if axis == 0:
        row_index = np.arange(input.shape[1 - axis])
        topk_index = np.argpartition(x, k, axis=axis)[0:k, :]
        topk_data = x[topk_index, row_index]

        topk_index_sort = np.argsort(topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:k, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(x.shape[1 - axis])[:, None]
        topk_index = np.argpartition(x, k, axis=axis)[:, 0:k]
        topk_data = x[column_index, topk_index]
        topk_data = -topk_data if largest else topk_data
        topk_index_sort = np.argsort(topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:k][column_index, topk_index_sort]

    return topk_data_sort, topk_index_sort