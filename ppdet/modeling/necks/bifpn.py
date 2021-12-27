# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division

import paddle
import paddle.nn as nn

# from ppdet.core.workspace import register

# __all__ = ['BiFPN']


class SeparableConv(nn.Layer):
    def __init__(self, in_c, out_c):
        super(SeparableConv, self).__init__()
        self.add_sublayer(
            "conv_dw",
            nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=3, padding="SAME", groups=in_c, bias_attr=False)
        )
        self.add_sublayer(
            "conv_pw",
            nn.Conv2D(
                in_channels=out_c, 
                out_channels=out_c, 
                kernel_size=1, 
                bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(0.))
            )
        )

        self.add_sublayer(
            "bn",
            nn.BatchNorm(out_c, 
                        momentum=0.997, 
                        epsilon=1e-04, 
                        param_attr=paddle.ParamAttr(
                            initializer=nn.initializer.Constant(1.0), regularizer=paddle.regularizer.L2Decay(0.)
                        ), 
                        bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(0.)))
        )
    
    def forward(self, x):
        return self.bn(self.conv_pw(self.conv_dw(x)))


class ConvBN(nn.Layer):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=1, 
                 stride=1, 
                 padding="SAME", 
                 weight_attr=None, 
                 bias_attr=None,
                 momentum=0.997,
                 epsilon=1e-04,):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_attr=weight_attr,
            bias_attr=bias_attr
        )

        self.bn = nn.BatchNorm(out_channels, 
                               momentum=momentum, 
                               epsilon=epsilon, 
                               param_attr=paddle.ParamAttr(
                                   initializer=nn.initializer.Constant(1.0), regularizer=paddle.regularizer.L2Decay(0.)
                               ), 
                               bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(0.)))
        
    def forward(self, x):
        return self.bn(self.conv(x))


class Fnode(nn.Layer):
    def __init__(self, idx=-1, n_w=2, **kwargs):
        super(Fnode, self).__init__()
        in_c = kwargs.get("in_channels")
        out_c = kwargs.get("out_channels")

        self.add_parameter(
                "combine_edge_weights", 
                self.create_parameter([n_w], attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0)))
            )

        if idx > -1:
            self.add_sublayer(
                "combine.resample.{}.conv".format(idx),
                ConvBN(in_channels=in_c, out_channels=out_c, bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(0.)))
            )

        self.add_sublayer("after_combine.conv", SeparableConv(in_c=out_c, out_c=out_c))
        self.idx = idx
        self.n_w = n_w

    def forward(self, feats):
        w = nn.ReLU()(getattr(self, "combine_edge_weights"))
        w = w / paddle.sum(w)

        if self.idx > -1:
            feats[0] = getattr(self, "combine.resample.{}.conv".format(self.idx))(feats[0])

        feat = feats[0] * w[0]
        for i in range(1, self.n_w):
            feat += feats[i] * w[i]

        feat = nn.Swish()(feat)
        feat = getattr(self, "after_combine.conv")(feat)

        return feat


class FPNCell(nn.Layer):
    def __init__(self, in_channels_list, out_channels, n_w_list, idx_list):
        super(FPNCell, self).__init__()
        for i in range(len(n_w_list)):

            if i > 0 and i < 6:
                self.add_sublayer(
                    "fnode.{}".format(i), 
                    Fnode(idx=idx_list[i-1], n_w=n_w_list[i], in_channels=in_channels_list[idx_list[i-1]-3], out_channels=out_channels)
                )
            else:
                self.add_sublayer(
                    "fnode.{}".format(i), 
                    Fnode(n_w=n_w_list[i], in_channels=out_channels, out_channels=out_channels)
                )

    def forward(self, feats):
        p3, p4, p5, p6, p7_lateral = feats
        p7_up = nn.Upsample(scale_factor=2, mode='nearest')(p7_lateral)
        p6_lateral = getattr(self, "fnode.0")([p6, p7_up])

        p6_up = nn.Upsample(scale_factor=2, mode='nearest')(p6_lateral)
        p5_lateral = getattr(self, "fnode.1")([p5, p6_up])

        p5_up = nn.Upsample(scale_factor=2, mode='nearest')(p5_lateral)
        p4_lateral = getattr(self, "fnode.2")([p4, p5_up])

        p4_up = nn.Upsample(scale_factor=2, mode='nearest')(p4_lateral)
        p3_final = getattr(self, "fnode.3")([p3, p4_up])

        p3_down = nn.MaxPool2D(kernel_size=3, stride=2, padding="SAME")(p3_final)
        p4_final = getattr(self, "fnode.4")([p4, p4_lateral, p3_down])

        p4_down = nn.MaxPool2D(kernel_size=3, stride=2, padding="SAME")(p4_final)
        p5_final = getattr(self, "fnode.5")([p5, p5_lateral, p4_down])

        p5_down = nn.MaxPool2D(kernel_size=3, stride=2, padding="SAME")(p5_final)
        p6_final = getattr(self, "fnode.6")([p6, p6_lateral, p5_down])

        p6_down = nn.MaxPool2D(kernel_size=3, stride=2, padding="SAME")(p6_final)
        p7_final = getattr(self, "fnode.7")([p7_lateral, p6_down])

        return [p3_final, p4_final, p5_final, p6_final, p7_final]


# @register
class BiFPN(nn.Layer):
    def __init__(self, in_c_list, out_c, num_r):
        super(BiFPN, self).__init__()
        n_w_list = [2, 2, 2, 2, 3, 3, 3, 2]
        idx_list = [2, 1, 0, 1, 2]

        self.level = len(idx_list)

        self.add_sublayer(
            "resample.{}.conv".format(self.level-2),
            ConvBN(in_channels=in_c_list[-1], out_channels=out_c, bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(0.)))
        )

        for i in range(num_r):
            if i == 0:
                self.add_sublayer(
                    "cell.{}".format(i),
                    FPNCell(in_c_list, out_c, n_w_list, idx_list)
                )
            else:
                self.add_sublayer(
                    "cell.{}".format(i),
                    FPNCell([out_c]*len(idx_list), out_c, n_w_list, [-1]*len(idx_list))
                )
        self.num_r = num_r

    def forward(self, inputs):
        c3, c4, c5 = inputs
        c6 = getattr(self, "resample.{}.conv".format(self.level-2))(c5)
        c6 = nn.MaxPool2D(kernel_size=3, stride=2, padding="SAME")(c6)
        c7 = nn.MaxPool2D(kernel_size=3, stride=2, padding="SAME")(c6)

        feats = [c3, c4, c5, c6, c7]

        for i in range(self.num_r):
            print(i)
            feats = getattr(self, "cell.{}".format(i))(feats)

        return feats

bifpn = BiFPN([40, 112, 320], 64, 3)

c3 = paddle.rand([2, 40, 400, 400])
c4 = paddle.rand([2, 112, 200, 200])
c5 = paddle.rand([2, 320, 100, 100])

out = bifpn([c3, c4, c5])

for i in out:
    print(i.shape)