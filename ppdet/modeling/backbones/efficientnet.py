from __future__ import absolute_import
from __future__ import division

import collections
import math
import re

import paddle
import paddle.nn as nn
from ppdet.core.workspace import register

__all__ = ['EfficientNet']

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'width_coefficient',
    'depth_coefficient', 'depth_divisor'
])

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'stride', 'se_ratio'
])

GlobalParams.__new__.__defaults__ = (None, ) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None, ) * len(BlockArgs._fields)


def _decode_block_string(block_string):
    assert isinstance(block_string, str)

    ops = block_string.split('_')
    options = {}
    for op in ops:
        splits = re.split(r'(\d.*)', op)
        if len(splits) >= 2:
            key, value = splits[:2]
            options[key] = value

    assert (('s' in options and len(options['s']) == 1) or
            (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

    return BlockArgs(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        se_ratio=float(options['se']) if 'se' in options else None,
        stride=int(options['s'][0]))


def get_model_params(scale):
    block_strings = [
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    block_args = []
    for block_string in block_strings:
        block_args.append(_decode_block_string(block_string))

    params_dict = {
        # width, depth
        'b0': (1.0, 1.0),
        'b1': (1.0, 1.1),
        'b2': (1.1, 1.2),
        'b3': (1.2, 1.4),
        'b4': (1.4, 1.8),
        'b5': (1.6, 2.2),
        'b6': (1.8, 2.6),
        'b7': (2.0, 3.1),
    }

    w, d = params_dict[scale]

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        width_coefficient=w,
        depth_coefficient=d,
        depth_divisor=8)

    return block_args, global_params


def round_filters(filters, global_params):
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    filters *= multiplier
    min_depth = divisor
    new_filters = max(min_depth,
                      int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def batch_norm(num_channels, momentum=0.9, eps=1e-05, norm_type="bn"):
    param_attr = paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(0.))
    bias_attr = paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(0.))

    if norm_type == "sync_bn":
        return nn.SyncBatchNorm(num_channels, weight_attr=param_attr, bias_attr=bias_attr)
    
    else:
        return nn.BatchNorm(num_channels, 
                            momentum=momentum, 
                            epsilon=eps, 
                            param_attr=param_attr, 
                            bias_attr=bias_attr, 
                            moving_mean_name=None, 
                            moving_variance_name=None)


class MBConvBlock(nn.Layer):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 expand_ratio,
                 kernel_size,
                 stride,
                 momentum,
                 eps,
                 se_ratio=None,
                 name=None):
        super(MBConvBlock, self).__init__()

        expand_channels = in_channels * expand_ratio
        self.num_bn = 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.se_ratio = se_ratio

        if expand_ratio != 1:
            self.add_sublayer("conv_pw", nn.Conv2D(in_channels, expand_channels, kernel_size=1, padding="SAME", bias_attr=False))
            self.add_sublayer("bn{}".format(self.num_bn), nn.BatchNorm(expand_channels, momentum=momentum, epsilon=eps))
            self.add_sublayer("swish{}".format(self.num_bn), nn.Swish())
            self.num_bn += 1

        self.add_sublayer("conv_dw", nn.Conv2D(expand_channels, 
                                               expand_channels, 
                                               kernel_size=kernel_size, 
                                               stride=stride, 
                                               groups=expand_channels, 
                                               padding="SAME", 
                                               bias_attr=False))
        self.add_sublayer("bn{}".format(self.num_bn), nn.BatchNorm(expand_channels, momentum=momentum, epsilon=eps))
        self.add_sublayer("swish{}".format(self.num_bn), nn.Swish())
        self.num_bn += 1

        self.se = nn.Sequential()
        if se_ratio is not None:
            squeezed_channels = max(1, int(in_channels * se_ratio))
            self.se.add_sublayer("global_avg_pool", nn.AdaptiveAvgPool2D(1))
            self.se.add_sublayer("conv_reduce", nn.Conv2D(expand_channels, squeezed_channels, kernel_size=1, padding="SAME"))
            self.se.add_sublayer("swish", nn.Swish())
            self.se.add_sublayer("conv_expand", nn.Conv2D(squeezed_channels, expand_channels, kernel_size=1, padding="SAME"))
            self.se.add_sublayer("sigmoid", nn.Sigmoid())

        self.name_pw = "conv_pwl" if expand_ratio != 1 else "conv_pw"
        self.add_sublayer(self.name_pw, nn.Conv2D(expand_channels, out_channels, kernel_size=1, padding="SAME", bias_attr=False))
        self.add_sublayer("bn{}".format(self.num_bn), nn.BatchNorm(out_channels, momentum=momentum, epsilon=eps))

    def forward(self, inputs):
        x = inputs
        if self.expand_ratio != 1:
            x = self.conv_pw(x)
            x = self.bn1(x)
            x = self.swish1(x)
        
        x = self.conv_dw(x)
        x = getattr(self, "bn{}".format(self.num_bn - 1))(x)
        x = getattr(self, "swish{}".format(self.num_bn - 1))(x)

        if self.se_ratio is not None:
            x_squeezed = self.se.global_avg_pool(x)
            x_squeezed = self.se.conv_reduce(x_squeezed)
            x_squeezed = self.se.swish(x_squeezed)
            x_squeezed = self.se.conv_expand(x_squeezed)
            x = x * self.se.sigmoid(x_squeezed)

        x = getattr(self, self.name_pw)(x)
        x = getattr(self, "bn{}".format(self.num_bn))(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            x = inputs + x

        return x


@register
class EfficientNet(nn.Layer):
    """
    EfficientNet, see https://arxiv.org/abs/1905.11946
    Args:
        scale (str): compounding scale factor, 'b0' - 'b7'.
        use_se (bool): use squeeze and excite module.
        norm_type (str): normalization type, 'bn' and 'sync_bn' are supported
    """
    def __init__(self, scale='b0', use_se=True, return_idx=[2, 4, 6], norm_type='bn'):
        super(EfficientNet, self).__init__()

        assert scale in ['b' + str(i) for i in range(8)], \
            "valid scales are b0 - b7"
        assert norm_type in ['bn', 'sync_bn'], \
            "only 'bn' and 'sync_bn' are supported"

        self.scale = scale
        self.use_se = use_se
        self.return_idx = return_idx
        self.norm_type = norm_type

        blocks_args, global_params = get_model_params(self.scale)
        momentum = global_params.batch_norm_momentum
        eps = global_params.batch_norm_epsilon
        num_filters = round_filters(32, global_params)

        self.stem = []
        self.add_sublayer("conv_stem", nn.Conv2D(3, num_filters, kernel_size=3, stride=2, padding="SAME", bias_attr=False))
        self.add_sublayer("bn1", batch_norm(num_filters, momentum, eps))
        self.stem_act = nn.Swish()
        self.stem.append(self.conv_stem)
        self.stem.append(self.bn1)
        self.stem.append(self.stem_act)

        self.blocks = []
        idx = -1
        self.feats_idx = []

        for i, block_arg in enumerate(blocks_args):
            for j in range(block_arg.num_repeat):
                in_channels = round_filters(block_arg.input_filters,
                                              global_params)
                out_channels = round_filters(block_arg.output_filters,
                                               global_params)
                kernel_size = block_arg.kernel_size
                stride = block_arg.stride
                se_ratio = None
                if self.use_se:
                    se_ratio = block_arg.se_ratio

                if j > 0:
                    in_channels = out_channels
                    stride = 1

                block = self.add_sublayer("blocks.{}.{}".format(i, j), MBConvBlock(
                    in_channels,
                    out_channels,
                    block_arg.expand_ratio,
                    kernel_size,
                    stride,
                    momentum,
                    eps,
                    se_ratio=se_ratio,
                ))
                self.blocks.append(block)
                idx += 1

            self.feats_idx.append(idx) 

    def forward(self, inputs):
        x = inputs
        for layer in self.stem:
            x = layer(x)

        feats = []

        for i, block in enumerate(self.blocks):
            x = block(x)

            if i in self.feats_idx:
                feats.append(x)
        
        return list(feats[i] for i in self.return_idx)
