import tensorflow as tf
# import tensorflow.keras as keras
from tensorflow.python.keras import Sequential
import tensorflow.keras.backend as K
import numpy as np
import tensorflow.keras.layers as layers
from model.layers.custom_layers import Resize, GroupNormalization


class MaskFeatHead(layers.Layer):
    def __init__(self,
                 in_channels,   
                 out_channels,  
                 start_level,   
                 end_level,     
                 nums,   
                 conv_cfg=None,   
                 norm_cfg=None):    
        super(MaskFeatHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.num_classes = nums
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.convs_all_levels = []
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = Sequential()
            if i == 0:
                one_conv = Sequential([
                    layers.Conv2D(
                    self.out_channels,
                    3,
                    padding='same'),

                    GroupNormalization(32),
                    
                    layers.ReLU()
                ])
                convs_per_level.add(one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    one_conv = Sequential([
                        layers.Conv2D(
                        self.out_channels,
                        3,
                        padding='same'),

                        GroupNormalization(32),

                        layers.ReLU()
                    ])
                    convs_per_level.add(one_conv)
                    one_upsample = layers.UpSampling2D(
                        size=(2,2), interpolation='bilinear')
                    convs_per_level.add(one_upsample)
                    continue

                one_conv = Sequential([
                    layers.Conv2D(
                    self.out_channels,
                    3,
                    padding='same'),

                    GroupNormalization(32),
                    layers.ReLU()
                ])
                convs_per_level.add(one_conv)
                one_upsample = layers.UpSampling2D(
                    size=(2,2), interpolation='bilinear')
                convs_per_level.add(one_upsample)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = Sequential([
            layers.Conv2D(
                self.num_classes,
                1,
                padding='valid'),

                GroupNormalization(32),

                layers.ReLU()
        ])

   
    def forward(self, inputs):
        assert len(inputs) == (self.end_level - self.start_level + 1)

        feature_add_all_level = self.convs_all_levels[0](inputs[0])
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == 3:
                input_feat = input_p
                x_range = tf.linspace(-1, 1, input_feat.shape[-1])
                y_range = tf.linspace(-1, 1, input_feat.shape[-2])
                y, x = tf.meshgrid(y_range, x_range)
                y = tf.broadcast_to(y, [input_feat.shape[0], 1, -1, -1])
                x = tf.broadcast_to(x, [input_feat.shape[0], 1, -1, -1])
                coord_feat = tf.cast(tf.concat([x, y], 1), dtype = tf.float32)
                input_p = tf.concat([input_p, coord_feat], 1)
                
            feature_add_all_level += self.convs_all_levels[i](input_p)

        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred
