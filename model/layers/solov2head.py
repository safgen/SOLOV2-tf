import tensorflow as tf
from model.layers.coordconv import CoordConv2D
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import layers
from data.imgutils import imrescale
from functools import partial
from six.moves import map, zip
from model.layers.custom_layers import GroupNormalization
from loss.loss import FocalLoss
import cv2
from scipy import ndimage

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

def dice_loss(y_pred, y_true):
    pq = tf.math.reduce_sum(tf.math.multiply(y_pred, y_true), axis=[1,2], keepdims=self.keepdims)
    p2 = tf.math.reduce_sum(tf.math.multiply(y_pred, y_pred), axis=[1,2], keepdims=self.keepdims)
    q2 = tf.math.reduce_sum(tf.math.multiply(y_true, y_true), axis=[1,2], keepdims=self.keepdims)
    return 1 - 2 * pq / (p2 + q2) 

class SOLOV2Head(tf.keras.layers.Layer):
    """
    Prediction head for SOLO V2 network
    """
    def __init__(self,
                 num_classes,  #2 
                 in_channels,  
                 seg_feat_channels=256,    
                 stacked_convs=4,        
                 strides=(4, 8, 16, 32, 64),  
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((56, 256)),
                 sigma=0.2,
                 num_grids=None,  
                 ins_out_channels=64,  
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
        self.strides = strides
        self.sigma = sigma
        self.stacked_convs = stacked_convs  #2
        self.kernel_out_channels = self.ins_out_channels * 1 * 1
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges

        self.loss_cate = FocalLoss(gamma=2.0,
            alpha=0.25,
            ) 
        
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.coordconv = CoordConv2D(filters=self.in_channels, kernel_size=(3,3), strides=1, padding="same",
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                    activation="relu")
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
                    chn,
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
                    chn ,
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
        # ins_kernel_feat = self.coordconv(x)
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
        
        cate_feat = kernel_feat

        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)    

        # cate branch
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)
       
        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return cate_pred, kernel_pred



    def loss(self,
             cate_preds,
             kernel_preds,
             ins_pred,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list,
             gt_bboxes_ignore=None):
         
        
        mask_feat_size = ins_pred.shape[-2:]
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = multi_apply(
            self.solov2_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list, 
            mask_feat_size=mask_feat_size)


        # ins
        ins_labels = [tf.concat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        kernel_preds = [[tf.reshape(kernel_preds_level_img,[kernel_preds_level_img.shape[0], -1])[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]
        # generate masks
        ins_pred = ins_pred
        ins_pred_list = []
        for b_kernel_pred in kernel_preds:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):

                if kernel_pred.shape()[-1] == 0:
                    continue
                cur_ins_pred = ins_pred[idx, ...]
                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape
                cur_ins_pred = tf.expand_dims(cur_ins_pred, axis=0)
                kernel_pred = tf.reshape(tf.transpose(kernel_pred, perm=[1,0]), [I, -1, 1, 1])
                cur_ins_pred = tf.reshape(tf.nn.conv2d(cur_ins_pred, kernel_pred, stride=1), [-1, H, W])
                b_mask_pred.append(cur_ins_pred)
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = tf.concat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)

        ins_ind_labels = [
            tf.concat([tf.layer.flatten(ins_ind_labels_level_img)
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = tf.concat(ins_ind_labels)

        num_ins = tf.reduce_sum(flatten_ins_ind_labels)
        # dice loss
        loss_ins = []
        for input, target in zip(ins_pred_list, ins_labels):
            if input is None:
                continue
            input = tf.nn.sigmoid(input)
            loss_ins.append(dice_loss(input, target))
        loss_ins = tf.reduce_mean(tf.concat(loss_ins))
        loss_ins = loss_ins * 3 #self.ins_loss_weight

        # cate
        cate_labels = [
            tf.concat([tf.layer.flatten(cate_labels_level_img)
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = tf.concat(cate_labels)

        cate_preds = [
            tf.reshape(tf.transpose(cate_pred, perm =[0, 2, 3, 1]),[-1, self.cate_out_channels])
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = tf.concat(cate_preds)
        loss_cate = self.loss_cate( flatten_cate_labels, flatten_cate_preds)
        total_loss = loss_cate + loss_ins
        return  total_loss, loss_cate, loss_ins

    def solov2_target_single(self,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               mask_feat_size):

        device = gt_labels_raw[0].device

        # ins
        gt_areas = tf.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.seg_num_grids):

            hit_indices = tf.flatten(((gt_areas >= lower_bound) & (gt_areas <= upper_bound)))
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            cate_label = tf.zeros([num_grid, num_grid], dtype=tf.dtypes.int64)
            ins_ind_label = tf.zeros([num_grid ** 2], dtype=tf.dtypes.bool)

            if num_ins == 0:
                ins_label = tf.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=tf.dtypes.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            output_stride = 4
            for seg_mask, gt_label, half_h, half_w in zip(gt_masks, gt_labels, half_hs, half_ws):
                if seg_mask.sum() == 0:
                   continue
                # mass center
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                cate_label[top:(down+1), left:(right+1)] = gt_label
                seg_mask = imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = tf.Tensor(seg_mask)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)

                        cur_ins_label = tf.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=tf.dtypes.uint8,
                                            )
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            ins_label = tf.stack(ins_label, 0)

            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

   

if __name__ == "__main__":
    sv2 = SOLOV2Head(2, 256)