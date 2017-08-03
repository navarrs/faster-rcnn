from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function 

import numpy as np 
import cv2 
import libs.boxes.cython as cython_bbox
import libs.configs.config_v1 as cfg
from libs.logs.log import LOG
from libs.boxes.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes

_DEBUG = False

def encode(gt_masks, gt_boxes, rois, num_classes, mask_height, mask_width):
    
    """ Encode mask ground truth into learnable targets 
        - Params: 
            gt_masks: im_h x im_w {0, 1} matrix, of shape (G, im_h, im_w)
            gt_boxes: of shape (G, 5), each raw is [x1, y1, x2, y2, class]
            rois: bounding boxes of shape (N, 4)
            ## scores of shape (N, 1)
            num_classes: k t
            mask_height, mask_width: of output masks
            
        - Return:
            # rois: boxes sample for cropping masks of shape (M, 4)
            labels: class-ids of shape (M, 1)
            mask_targets: learning targets of shape (M, pooled_height, pooled_width, k) in [0, 1] vals
            mask_inside_weights: of shape (M, pooled_height, pooled_width, K) in [0, 1] indicating with mask is sampled
    """
    
    total_masks = rois.shape[0]
    
    if gt_boxes.size > 0:
        # B x G
        overlaps = cython_bbox.bbox_overlaps(
            np.ascontiguousarray(rois[:, 0:4], dtype=np.float), 
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        
        gt_assignment = overlaps.argmax(axis=1) # shape is N
        max_overlaps = overlaps[np.arange(len(gt_assignment)), gt_assignment] # N 
        # note: this will assign every rois with a positive label 
        # labels = gt_boxes[gt_assignment, 4]  # N
        labels = np.zeros((total_masks, ), np.float32)
        labels[:] = -1
        
        # sample positive rois which intersection is more than 0.5
        keep_inds = np.where(max_overlaps >= cfg.FLAGS.mask_threshold)[0]
        num_masks = int(min(keep_inds.size, cfg.FLAGS.mask_per_image))
        if keep_inds.size > 0 and num_masks < keep_inds.size:
            keep_inds = np.random.choice(keep_inds, size=num_masks, replace=False)
            LOG('Mask: %d of %d rois are considered positive mask. Number of mask %d' \
                %(num_masks, rois.shape[0], gt_masks.shape[0]))
        
        labels[keep_inds] = gt_boxes[gt_assignment[keep_inds], -1]
        
        # rois = rois[inds]
        # labels = labels[inds].astype(np.int32) 
        # gt_assignment = gt_assignment[inds]
        
        # ignore rois with overlaps between fg_threshold and bg_threshold
        # mask are only defined on positive axis 
        ignore_inds = np.where((max_overlaps < cfg.FLAGS.fg_threshold))[0]
        labels[ignore_inds] = -1 
        
        mask_targets = np.zeros((total_masks, mask_height, mask_width, num_classes), dtype=np.int32)
        mask_inside_weights = np.zeros((total_masks, mask_height, mask_width, num_classes), dtype=np.float32)
        rois[rois < 0] = 0
        
        # TODO: speed bottleneck 
        for i in keep_inds:
            roi = rois[i, :4]
            
            cropped = gt_masks[gt_assignment[i], int(roi[i]):int(roi[3])+1, int(roi[0]):int(roi[2])+1]
            cropped = cv2.resize(cropped, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)
            
            mask_targets[i, :, :, int(labels[i])] = cropped
            mask_inside_weights[i, :, :, int(labels[i])] = 1
            
    else:
        # there is np gt
        

        