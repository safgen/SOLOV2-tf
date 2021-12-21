import tensorflow as tf
# from model.layers.coordconv import CoordConv2D
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import layers
import tensorflow_addons as tfa
from functools import partial
from six.moves import map, zip
from model.layers.custom_layers import Resize, GroupNormalization


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
            map the multiple outputs of the ``func`` into different
            list. Each list contains the same type of outputs corresponding
            to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = tf.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

class SOLOV2Head(tf.keras.layers.Layer):
    """
    Prediction head for SOLO V2 network
    """
    def __init__(self,
                 num_classes,  #2 
                 in_channels,  # 256 fpn outputs
                 seg_feat_channels=256,   #seg feature channels 
                 stacked_convs=4,        #solov2 light set 2
                 strides=(4, 8, 16, 32, 64),  # [8, 8, 16, 32, 32],
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((56, 256)),
                 sigma=0.2,
                 num_grids=None,  #[40, 36, 24, 16, 12],
                 ins_out_channels=64,  #128
                 loss_ins=None,
                 loss_cate=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SOLOV2Head, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        # self.strides = strides
        self.sigma = sigma
        self.stacked_convs = stacked_convs  #2
        self.kernel_out_channels = self.ins_out_channels * 1 * 1
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.cate_convs = []
        self.kernel_convs = []
        for i in range(self.stacked_convs):   
            # conv2d_cate = layers.Conv2D(self.seg_feat_channels, 3, padding='same', strides=1, use_bias=False)
            # # gn_c = GroupNormalization(num_groups=32)
            # relu_c = layers.ReLU()
            # self.cate_convs.append(conv2d_cate)
            # # self.cate_convs.append(gn_c)
            # self.cate_convs.append(relu_c)

            # conv2d_k = layers.Conv2D(self.seg_feat_channels, 3, padding='same', strides=1, use_bias=False)
            # # gn_k = GroupNormalization(num_groups=32)
            # relu_k = layers.ReLU()
            # self.cate_convs.append(conv2d_k)
            # # self.cate_convs.append(gn_k)
            # self.cate_convs.append(relu_k)

       
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(Sequential([
                layers.Conv2D(
                    self.seg_feat_channels,
                    3,
                    strides=1,
                    padding='same'
                    # bias=norm_cfg is None
                    ),

                # tfa.layers.GroupNormalization(groups=32), #TODO fix
                # tfa.layers.GroupNormalization(),
                GroupNormalization(num_groups=32),
                    
                layers.ReLU()
            ]))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(Sequential([
                layers.Conv2D(
                    self.seg_feat_channels,
                    3,
                    strides=1,
                    padding='same'
                    # bias=norm_cfg is None
                    ),

                # tfa.layers.GroupNormalization(groups=32),
                GroupNormalization(num_groups=32),
                    
                layers.ReLU()
            ]))

        self.solo_cate = layers.Conv2D(self.cate_out_channels, 3, padding='same')

        self.solo_kernel = layers.Conv2D(self.kernel_out_channels, 3, padding='same')

    
    def call(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        print(new_feats)
        featmap_sizes = [featmap.shape[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        cate_pred, kernel_pred = multi_apply(self.forward_single, new_feats,
                                                       list(range(len(self.seg_num_grids))),
                                                       eval=eval, upsampled_size=upsampled_size)
        return cate_pred, kernel_pred
    
    def split_feats(self, feats):
        return (
                tf.image.resize(feats[0], feats[1].shape[-2:]),
                feats[1],
                feats[2],
                feats[3],
                tf.image.resize(feats[4], feats[2].shape[-2:])) #bilinear default
               
    def forward_single(self, x, idx, eval=False, upsampled_size=None):
        ins_kernel_feat = x
        # ins branch
        # concat coord
        x_range = tf.linspace(-1, 1, ins_kernel_feat.shape[-1], name='x-linspace')
        y_range = tf.linspace(-1, 1, ins_kernel_feat.shape[-2], name='y-linspace')
        y, x = tf.meshgrid(y_range, x_range)
        y = tf.broadcast_to(y, [ins_kernel_feat.shape[0], 1, -1, -1])
        x = tf.broadcast_to(x, [ins_kernel_feat.shape[0], 1, -1, -1])
        coord_feat = tf.cast(tf.concat([x, y], 1), dtype = tf.float32)
        ins_kernel_feat = tf.concat([ins_kernel_feat, coord_feat], 1)
        
        
        # kernel branch
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]  
        kernel_feat = tf.image.resize(kernel_feat, [seg_num_grid, seg_num_grid])
        
        cate_feat = kernel_feat[:, :-2, :, :]

        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)    #TODO 

        # cate branch
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)

        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return cate_pred, kernel_pred


   

if __name__ == "__main__":
    sv2 = SOLOV2Head(2, 256)