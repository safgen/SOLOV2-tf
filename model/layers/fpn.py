import tensorflow as tf
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import ResizeMethod
from tensorflow.keras import layers
from model.layers.custom_layers import Conv2dUnit


class FPN(tf.keras.layers.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(FPN, self).__init__()

        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.add_extra_convs = add_extra_convs
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level


        self.lateral_convs = []
        self.fpn_convs = []
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = Conv2dUnit(out_channels, 1, strides=1, padding='valid', use_bias=True, bn=0, act=None)
            fpn_conv = Conv2dUnit(out_channels, 3, strides=1, padding='same', use_bias=True, bn=0, act=None)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def __call__(self, xs):
        num_ins = len(xs)

        # build laterals
        laterals = []
        for i in range(num_ins):
            x = self.lateral_convs[i](xs[i + self.start_level])
            laterals.append(x)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            x = layers.UpSampling2D(2)(laterals[i])
            laterals[i - 1] = layers.add([laterals[i - 1], x])

        # build outputs
        # part 1: from original levels
        outs = []
        for i in range(used_backbone_levels):
            x = self.fpn_convs[i](laterals[i])
            outs.append(x)
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    x = layers.MaxPooling2D(pool_size=1, strides=2, padding='valid')(outs[-1])
                    outs.append(x)
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                pass
        return outs
