'''
    Created on: 06.12.2021
    @Author: feizzhang
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

from ppdet.modeling.proposal_generator.target import label_box
from ppdet.modeling.bbox_utils import bbox2delta

__all__ = ["GHMLoss"]

@register
class GHMLoss(nn.Layer):
    def __init__(
        self,
        positive_thresh,
        negative_thresh,
        allow_low_quality=True,
        ghm_c_bins=30,
        ghm_c_momentum=0.75,
        ghm_c_use_sigmoid=True,
        ghm_c_weight=1.0,
        ghm_c_reduction='mean',
        ghm_r_mu=0.02,
        ghm_r_bins=10,
        ghm_r_momentum=0.7,
        ghm_r_weight=10.0,
        ghm_r_reduction="mean",
        num_classes=80,
        weights=[1.0, 1.0, 1.0, 1.0]
    ):
        super(GHMLoss, self).__init__()
        self.cls_loss = GHMC(
            bins=ghm_c_bins,
            momentum=ghm_c_momentum,
            use_sigmoid=ghm_c_use_sigmoid,
            loss_weight=ghm_c_weight,
            reduction=ghm_c_reduction
        )
        self.reg_loss = GHMR(
            mu=ghm_r_mu,
            bins=ghm_r_bins,
            momentum=ghm_r_momentum,
            loss_weight=ghm_r_weight,
            reduction=ghm_r_reduction
        )
        self.num_classes = num_classes
        self.positive_thresh = positive_thresh
        self.negative_thresh = negative_thresh
        self.allow_low_quality = allow_low_quality
        self.weights = weights

    def label_anchors(self, anchors, gt):
        batch_gt_box = gt["gt_bbox"]
        batch_gt_class = gt["gt_class"]

        gt_labels_list = []
        gt_boxes_list = []

        for i in range(len(batch_gt_box)):
            gt_boxes = batch_gt_box[i]
            gt_classes = batch_gt_class[i].flatten()
            matches_idxs, match_labels = label_box(anchors,
                                                   gt_boxes,
                                                   self.positive_thresh,
                                                   self.negative_thresh,
                                                   self.allow_low_quality,
                                                   -1)

            if len(gt_boxes) > 0:
                matched_boxes_i = paddle.gather(gt_boxes, matches_idxs)
                matched_classes_i = paddle.gather(gt_classes, matches_idxs)
                matched_classes_i = paddle.where(match_labels == 0,
                                                 paddle.full_like(matched_classes_i, self.num_classes),
                                                 matched_classes_i)
                matched_classes_i = paddle.where(match_labels == -1,
                                                 paddle.full_like(matched_classes_i, -1),
                                                 matched_classes_i)
            else:
                matched_boxes_i = paddle.zeros_like(anchors)
                matched_classes_i = paddle.zeros_like(matches_idxs) + self.num_classes

            gt_boxes_list.append(matched_boxes_i)
            gt_labels_list.append(matched_classes_i)
        
        return gt_boxes_list, gt_labels_list

    def forward(self, anchors, preds, inputs):

        pred_scores_list, pred_boxes_list = preds

        p_s = paddle.concat(pred_scores_list, axis=1)
        p_b = paddle.concat(pred_boxes_list, axis=1)  # [N, R, 4]

        gt_boxes, gt_classes = self.label_anchors(anchors, inputs)
        gt_labels = paddle.stack(gt_classes).reshape([-1])  # [N * R]

        valid_idx = paddle.nonzero(gt_labels >= 0)
        pos_mask = paddle.logical_and(gt_labels >= 0, gt_labels != self.num_classes)
        pos_idx = paddle.nonzero(pos_mask).flatten()

        p_s = paddle.reshape(p_s, [-1, self.num_classes])
        pred_logits = paddle.gather(p_s, valid_idx)

        gt_labels = F.one_hot(paddle.gather(gt_labels, valid_idx), num_classes=self.num_classes + 1)[
            :, :-1
        ]
        
        gt_labels.stop_gradient = True

        cls_loss = self.cls_loss(pred_logits, gt_labels)

        gt_deltas_list = [
            bbox2delta(anchors, gt_boxes[i], self.weights) for i in range(len(gt_boxes))
        ]

        gt_deltas = paddle.concat(gt_deltas_list)
        gt_deltas = paddle.gather(gt_deltas, pos_idx)
        gt_deltas.stop_gradient = True

        p_b = paddle.reshape(p_b, [-1, 4])
        pred_deltas = paddle.gather(p_b, pos_idx)

        reg_loss = self.reg_loss(pred_deltas, gt_deltas)

        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss
        }


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


class GHMC(nn.Layer):
    """GHM Classification Loss.
    Details of the theorem can be viewed in the paper
    `Gradient Harmonized Single-stage Detector
    <https://arxiv.org/abs/1811.05181>`_.
    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
        reduction (str): Options are "none", "mean" and "sum".
            Defaults to "mean"
    """

    def __init__(self,
                 bins=10,
                 momentum=0,
                 use_sigmoid=True,
                 loss_weight=1.0,
                 reduction='mean'):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = paddle.arange(bins + 1).astype("float32") / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = paddle.zeros([bins]).astype("float32")
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,
                pred,
                target,
                reduction_override=None,
                **kwargs):
        """Calculate the GHM-C loss.
        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            The gradient harmonized loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction
        )

        edges = self.edges
        mmt = self.momentum
        weights = paddle.zeros_like(pred)

        # gradient length
        g = paddle.abs(F.sigmoid(pred) - target)
        g.stop_gradient = True

        # valid = label_weight > 0
        tot = max(float(target.size), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            # inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            mask = paddle.logical_and(g >= edges[i], g < edges[i + 1])
            num_in_bin = mask.astype("float32").sum()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin

                    weights[mask] = tot / self.acc_sum[i]
                else:
                    weights[mask] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n
        # print(weights)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none')
        loss = weight_reduce_loss(
            loss, weights, reduction=reduction, avg_factor=tot)
        return loss * self.loss_weight


class GHMR(nn.Layer):
    """GHM Regression Loss.
    Details of the theorem can be viewed in the paper
    `Gradient Harmonized Single-stage Detector
    <https://arxiv.org/abs/1811.05181>`_.
    Args:
        mu (float): The parameter for the Authentic Smooth L1 loss.
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        loss_weight (float): The weight of the total GHM-R loss.
        reduction (str): Options are "none", "mean" and "sum".
            Defaults to "mean"
    """

    def __init__(self,
                 mu=0.02,
                 bins=10,
                 momentum=0,
                 loss_weight=1.0,
                 reduction='mean'):
        super(GHMR, self).__init__()
        self.mu = mu
        self.bins = bins
        edges = paddle.arange(bins + 1).astype("float32") / bins
        self.register_buffer('edges', edges)
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            acc_sum = paddle.zeros([bins]).astype("float32")
            self.register_buffer('acc_sum', acc_sum)
        self.loss_weight = loss_weight
        self.reduction = reduction

    # TODO: support reduction parameter
    def forward(self,
                pred,
                target,
                reduction_override=None):
        """Calculate the GHM-R loss.
        Args:
            pred (float tensor of size [batch_num, 4 (* class_num)]):
                The prediction of box regression layer. Channel number can be 4
                or 4 * class_num depending on whether it is class-agnostic.
            target (float tensor of size [batch_num, 4 (* class_num)]):
                The target regression values with the same size of pred.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            The gradient harmonized loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = pred - target
        loss = paddle.sqrt(diff * diff + mu * mu) - mu

        # gradient length
        g = paddle.abs(diff / paddle.sqrt(mu * mu + diff * diff))
        g.stop_gradient = True
        weights = paddle.zeros_like(g)

        tot = max(float(target.size), 1.0)
        n = 0  # n: valid bins
        for i in range(self.bins):
            mask = paddle.logical_and(g >= edges[i], g < edges[i + 1])
            num_in_bin = mask.astype("float32").sum()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[mask] = tot / self.acc_sum[i]
                else:
                    weights[mask] = tot / num_in_bin

        if n > 0:
            weights /= n

        loss = weight_reduce_loss(
            loss, weights, reduction=reduction, avg_factor=tot)
        return loss * self.loss_weight
