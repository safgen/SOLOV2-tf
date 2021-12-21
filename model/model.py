import tensorflow as tf
from model.layers.fpn import FPN
from model.layers.head import PredictionHead
from model.layers.solov2head import SOLOV2Head 
from model.layers.mask_head import MaskFeatHead
assert tf.__version__.startswith('2')


class SOLO(tf.keras.Model):
    def __init__(self, num_class, input_size, grid_sizes=[24], backbone="resnet50", head_style="vanilla", head_depth=8, fpn_channel=256, **kwargs):
        super(SOLO, self).__init__(**kwargs)
        self.num_class = num_class
        self.input_size = input_size
        self.grid_sizes = grid_sizes
        self.backbone_name = backbone
        self.head_style = head_style
        self.head_depth = head_depth
        self.fpn_channel = fpn_channel
        self.model_name = "SOLO_" + backbone

        if backbone == 'resnet50':
            self.backbone_out_layers = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
            base_model = tf.keras.applications.ResNet50(input_shape=(input_size, input_size, 3),
                                                        include_top=False,
                                                        layers=tf.keras.layers,
                                                        weights='imagenet')

        else:
            raise NotImplementedError('Backbone %s not supported' % (backbone))

        self.backbone = tf.keras.Model(inputs=base_model.input,
                                       outputs=[base_model.get_layer(x).output for x in self.backbone_out_layers], name=backbone)
        self.fpn = FPN(in_channels=[64, 128, 256, 512], out_channels=256, num_outs=5)
        self.mask_head = MaskFeatHead(in_channels=256,
                            out_channels=128,
                            start_level=0,
                            end_level=3,
                            nums=128)
        self.solo_head = SOLOV2Head(num_classes=2,
                            in_channels=256,
                            seg_feat_channels=256,
                            stacked_convs=4,
                            strides=[8, 8, 16, 32, 32],
                            scale_ranges=((1, 56), (28, 112), (56, 224), (112, 448), (224, 896)),
                            num_grids=grid_sizes,
                            ins_out_channels=128
                        )
        # self.bbox_head = DecoupledSOLOHead()


    def call(self, input, gt_bboxes, gt_labels, gt_masks, training=False):
        c2, c3, c4, c5 = self.backbone(input, training=training)

        Px = self.fpn([c2, c3, c4, c5])
        
        cate_out, kern_out = self.solo_head(Px, None)
        mask_out = self.mask_head(Px[self.mask_head.
                  start_level:self.mask_head.end_level + 1])

        loss_inputs = cate_out, kern_out, mask_out, gt_bboxes, gt_labels, gt_masks
        # print(loss_inputs.shape   )
        losses = self.solo_head.loss(
            *loss_inputs)
        return losses

        # cate = []
        # kern = []
        # for cat, k in zip(cate_out, kern_out):
        #     cate.append(cat)
        #     kern.append(k)

        return cate_out, kern_out, mask_out 
        
      

    def get_config(self):
        config = super(SOLO, self).get_config()
        config['num_class'] = self.num_class,
        config['input_size'] = self.input_size,
        config['grid_sizes'] = self.grid_sizes,
        config['backbone_name'] = self.backbone_name,
        config['backbone_out_layers'] = self.backbone_out_layers,
        config['head_style'] = self.head_style,
        config['head_depth'] = self.head_depth,
        config['fpn_channel'] = self.fpn_channel,
        config['model_name'] = self.model_name
        return config
