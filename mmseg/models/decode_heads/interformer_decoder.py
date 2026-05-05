# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from typing import Optional
import numpy as np
import cv2
from PIL import Image

try:
    from mmdet.models.dense_heads import \
        Mask2FormerHead as MMDET_Mask2FormerHead
except ModuleNotFoundError:
    MMDET_Mask2FormerHead = BaseModule

from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList

# moving from parent classes
from mmseg.utils import OptConfigType, OptMultiConfig
import copy
from mmdet.models.layers import Mask2FormerTransformerDecoder, SinePositionalEncoding
from mmseg.registry import TASK_UTILS
from mmengine.model import ModuleList, caffe2_xavier_init
from mmcv.cnn import Conv2d
from .decode_head import BaseDecodeHead
from mmdet.models.utils import multi_apply, get_uncertain_point_coords_with_randomness
from mmcv.ops import point_sample
from mmdet.utils import reduce_mean, InstanceList
from typing import Union


class UNetDecoder(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(UNetDecoder, self).__init__()
        
        self.upconv1 = nn.ConvTranspose2d(in_channels_list[3], in_channels_list[2], kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels_list[2] * 2, in_channels_list[2], kernel_size=3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(in_channels_list[2], in_channels_list[1], kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels_list[1] * 2, in_channels_list[1], kernel_size=3, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(in_channels_list[1], in_channels_list[0], kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels_list[0] * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        x1 = features[3] 
        x2 = self.upconv1(x1)  
        x2 = torch.cat((x2, features[2]), dim=1)  # Concatenate
        x2 = self.conv1(x2)
        
        x3 = self.upconv2(x2)  
        x3 = torch.cat((x3, features[1]), dim=1)  # Concatenate
        x3 = self.conv2(x3)
        
        x4 = self.upconv3(x3) 
        x4 = torch.cat((x4, features[0]), dim=1)  # Concatenate
        x4 = self.conv3(x4)
        
        return x4, [x1,x2,x3]
    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
    
class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0,
                 activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
     
        return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)

class Attention(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)


    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x, attn
    
def select_top_k_features_from_regional_similarity(feature_A: torch.Tensor, feature_B: torch.Tensor, k: int = 5):
    
    if feature_A.shape[1] != feature_B.shape[1]:
        raise ValueError("Not match")

    batch_size, channels, h_A, w_A = feature_A.shape
    _, _, h_B, w_B = feature_B.shape

    if h_B != 2 * h_A or w_B != 2 * w_A:
        raise ValueError("2x times Error")

    b_top_left = feature_B[:, :, :h_A, :w_A]
    b_top_right = feature_B[:, :, :h_A, w_A:]
    b_bottom_left = feature_B[:, :, h_A:, :w_A]
    b_bottom_right = feature_B[:, :, h_A:, w_A:]

    b_sub_regions = [b_top_left, b_top_right, b_bottom_left, b_bottom_right]

    pixelwise_similarities_list = []
    for b_sub in b_sub_regions:
     
        cosine_sim_map = F.cosine_similarity(feature_A, b_sub, dim=1)
        pixelwise_similarities_list.append(cosine_sim_map)

   
    sim_map_top = torch.cat([pixelwise_similarities_list[0], pixelwise_similarities_list[1]], dim=2) # [B, H_A, 2*W_A] = [B, H_A, W_B]
    sim_map_bottom = torch.cat([pixelwise_similarities_list[2], pixelwise_similarities_list[3]], dim=2) # [B, H_A, 2*W_A] = [B, H_A, W_B]
    
    total_similarity_map = torch.cat([sim_map_top, sim_map_bottom], dim=1) # [B, 2*H_A, W_B] = [B, H_B, W_B]

    final_selected_features = []

    for b_idx in range(batch_size): 
        
        batch_similarity = total_similarity_map[b_idx].flatten()

        _, top_k_indices_flat = torch.topk(batch_similarity, k=k, largest=True, sorted=True)

        current_batch_top_k_features = []
        for k_idx in range(k):
            flat_idx = top_k_indices_flat[k_idx].item()
        
            h_coord = flat_idx // w_B
            w_coord = flat_idx % w_B
            
            selected_feature = feature_B[b_idx, :, h_coord, w_coord].unsqueeze(-1).unsqueeze(-1)
            current_batch_top_k_features.append(selected_feature)
    
    all_batches_top_k_flat_indices = []
    for b_idx in range(batch_size):
        batch_similarity = total_similarity_map[b_idx].flatten()
        _, top_k_indices_flat = torch.topk(batch_similarity, k=k, largest=True, sorted=True)
        all_batches_top_k_flat_indices.append(top_k_indices_flat)

    final_selected_features_list = [] 

    for k_val_idx in range(k):
        features_for_this_k_across_batches = []
        for b_idx in range(batch_size): 
            flat_idx = all_batches_top_k_flat_indices[b_idx][k_val_idx].item()
            h_coord = flat_idx // w_B
            w_coord = flat_idx % w_B
            
            selected_feature = feature_B[b_idx, :, h_coord, w_coord].unsqueeze(-1).unsqueeze(-1)
            features_for_this_k_across_batches.append(selected_feature)
        
        final_selected_features_list.append(torch.stack(features_for_this_k_across_batches, dim=0))

    return final_selected_features_list
    
    
@MODELS.register_module()
class InterFormerDecoder(BaseDecodeHead):


    def __init__(self,
                 in_channels: List[int],
                 feat_channels: int,
                 out_channels: int,
                 num_queries: int = 100,
                 num_transformer_feat_level: int = 3,
                 pixel_decoder: ConfigType = ...,
                 enforce_decoder_input_project: bool = False,
                 transformer_decoder: ConfigType = ...,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=2.0,
                     reduction='mean',
                     class_weight=[1.0] * 133 + [0.1]),
                 loss_mask: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=5.0),
                 loss_dice: ConfigType = dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     reduction='mean',
                     naive_dice=True,
                     eps=1.0,
                     loss_weight=5.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 num_classes=10,
                 align_corners=False,
                 ignore_index=255,
                 **kwargs):
        super().__init__(in_channels=3,channels=out_channels,num_classes=num_classes)

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index

        feat_channels = feat_channels


        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers

        assert pixel_decoder.encoder.layer_cfg. \
            self_attn_cfg.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = MODELS.build(pixel_decoder_)
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            **transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = SinePositionalEncoding(
            **positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels) #position
        self.query_feat = nn.Embedding(self.num_queries, feat_channels) #content
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))
        
        
        self.mlp = nn.Sequential(nn.Linear(112*112, 256), nn.ReLU(inplace=True),
                                nn.Linear(256,128), nn.ReLU(inplace=True),
                                nn.Linear(128,5),nn.ReLU(inplace=True))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)
        

        self.class_weight = loss_cls.class_weight
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)


        self.cbdecoder = UNetDecoder(in_channels_list=[128, 256, 512, 1024], out_channels=1)
        self.proj_cb = ModuleList()
        channels = [1024,512,256]
        for i in range(num_transformer_feat_level):
            self.proj_cb.append(
                    Conv2d(
                        channels[i], self.decoder_embed_dims, kernel_size=1))
        self.cross_attn_cb = CrossAttentionLayer()
        self.self_attn_cb = Attention()
        self.norm_cb = nn.LayerNorm(256)
       
    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):

        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False) 
         
            gt_labels = classes[classes != self.ignore_index]

            masks = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros(
                    (0, gt_sem_seg.shape[-2],
                     gt_sem_seg.shape[-1])).to(gt_sem_seg).long()
            else:
                gt_masks = torch.stack(masks).squeeze(1).long()

            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas
    
    def compute_loss_for_cb(self, cb_pred, batch_data_samples):
        # add supervision for cb prediction
        gt_cbs = []
        for data_sample in batch_data_samples:
            img_path = data_sample.metainfo['img_path']
            filename = img_path.split('.')[0].split('/')[-1]
            cb_gt_path = '/mnt/nvme1/suyuejiao/egohos_split_data/train/label_contact_first/'+filename+'.png'
            cb_gt_np = np.array(Image.open(cb_gt_path).resize((448, 448)))
            gt_cbs.append(torch.from_numpy(cb_gt_np).to(cb_pred))

        gt_cbs = torch.stack(gt_cbs).squeeze(1).float()
      
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(cb_pred.squeeze(1), gt_cbs)
        return loss



    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
  
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)
        
        all_cls_scores, all_mask_preds, cb_pred = self.forward(x, batch_data_samples)

        # add loss for cb prediction
        loss_cb = self.compute_loss_for_cb(cb_pred,batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)
        losses['cb']=loss_cb

        # adding the consistency loss
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        pred_results_index = torch.argmax(seg_logits,dim=1)
      
        left_hand_pred = torch.zeros(pred_results_index.shape)
        left_hand_pred[pred_results_index==1] = 1
      
        threshold_left = 150
        if left_hand_pred.sum() >= threshold_left:
            index_left = 1
        else:
            index_left = 0
        
        right_hand_pred = torch.zeros(pred_results_index.shape)
        right_hand_pred[pred_results_index==2] = 1

        threshold_right = threshold_left
        if right_hand_pred.sum() >= threshold_right:
            index_right = 1
        else:
            index_right = 0
        
        if index_left == 0 or index_right == 0:
            index_two = 0
        else:
            index_two = 1

        left_object_pred = torch.zeros(pred_results_index.shape)
        left_object_pred[pred_results_index==3] = 1
        right_object_pred = torch.zeros(pred_results_index.shape)
        right_object_pred[pred_results_index==4] = 1
        two_object_pred = torch.zeros(pred_results_index.shape)
        two_object_pred[pred_results_index==5] = 1
        loss_left = (1-index_left)*(left_object_pred.sum()-left_object_pred.sum()*index_left)
        loss_right = (1-index_right)*(right_object_pred.sum()-right_object_pred.sum()*index_right)
        loss_two = (1-index_two)*(two_object_pred.sum()-two_object_pred.sum()*index_two)
        consistency_loss = loss_left + loss_right + loss_two

        losses['consistency'] = consistency_loss
        
        return losses
    
    def loss_by_feat(self, all_cls_scores: Tensor, all_mask_preds: Tensor,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
     
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice = multi_apply(
            self._loss_by_feat_single, all_cls_scores, all_mask_preds,
            batch_gt_instances_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict
    
    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
       
        labels = torch.stack(labels_list, dim=0)
   
        label_weights = torch.stack(label_weights_list, dim=0)

        mask_targets = torch.cat(mask_targets_list, dim=0)
      
        mask_weights = torch.stack(mask_weights_list, dim=0)

        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice
    
    def get_targets(
        self,
        cls_scores_list: List[Tensor],
        mask_preds_list: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        return_sampling_results: bool = False
    ) -> Tuple[List[Union[Tensor, int]]]:
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.
            return_sampling_results (bool): Whether to return the sampling
                results. Defaults to False.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - avg_factor (int): Average factor that is used to average\
                    the loss. When using sampling method, avg_factor is
                    usually the sum of positive and negative priors. When
                    using `MaskPseudoSampler`, `avg_factor` is usually equal
                    to the number of positive priors.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end.
        """
        results = multi_apply(self._get_targets_single, cls_scores_list,
                              mask_preds_list, batch_gt_instances,
                              batch_img_metas)
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])

        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])

        res = (labels_list, label_weights_list, mask_targets_list,
               mask_weights_list, avg_factor)
        if return_sampling_results:
            res = res + (sampling_results_list)

        return res + tuple(rest_results)
    
    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> Tuple[Tensor]:
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_instances (:obj:`InstanceData`): It contains ``labels`` and
                ``masks``.
            img_meta (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result)
    
    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """
        
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (batch_size,num_queries,  c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (batch_size,num_queries,  num_classes)
        mask_embed = self.mask_embed(decoder_out)
        # shape ( batch_size, num_queries, c)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask
    
    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        batch_size = x[0].shape[0]
        
        mask_features, multi_scale_memorys = self.pixel_decoder(x)


        # add decoder for CB prediction
        cb_feat, cb_multi_sclae_features = self.cbdecoder(x)
        cb_pred = F.interpolate(
            cb_feat, size=(448,448), mode='bilinear', align_corners=False)


        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            
            
            cb_feat_align_channel = self.proj_cb[i](cb_multi_sclae_features[i])
          
            cb_feat_align_channel = cb_feat_align_channel.flatten(2).permute(0, 2, 1)
     
            cb_cross_attn = self.cross_attn_cb(cb_feat_align_channel, decoder_input)
            cb_cross_attn = cb_cross_attn + cb_feat_align_channel
            cb_cross_attn = self.norm_cb(cb_cross_attn)
            cb_attn, A = self.self_attn_cb(cb_cross_attn)
           
            cb_attn = cb_cross_attn + cb_attn
            cb_attn = self.norm_cb(cb_attn)
            decoder_input = decoder_input+cb_attn

            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
       

 
        selected_q = select_top_k_features_from_regional_similarity(cb_multi_sclae_features[2], mask_features)
      
        _selected_q = [feat.squeeze().unsqueeze(-1) for feat in selected_q]
        selected_q = torch.cat(_selected_q, dim=2)
 
        selected_q = selected_q.permute(0,2,1)
    
       
        query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1)) 
        query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))
        query_feat = query_feat + selected_q

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)

            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:])
            
          

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list, cb_pred

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.
        """
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds, _ = self.forward(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1] 
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits
