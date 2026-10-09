"""
Module: metrics.py
Description: Core evaluation metrics and post-processing utilities for stroke 
             lesion segmentation. Centralizes Dice, ASD, and F1 calculations.
"""

import numpy as np
from scipy.ndimage import label, binary_fill_holes
from scipy.spatial import cKDTree
from skimage.morphology import remove_small_objects

def calculate_dice(p_array: np.ndarray, g_array: np.ndarray) -> float:
    """
    Computes the Global Dice Coefficient between the prediction and ground truth.
    """
    dice_score = 0.0
    sum_p = np.sum(p_array)
    sum_g = np.sum(g_array)
    
    if sum_p + sum_g == 0:
        dice_score = 1.0
    else:
        dice_score = 2.0 * np.sum(p_array * g_array) / (sum_p + sum_g)
        
    return dice_score

def calculate_asd(p_array: np.ndarray, g_array: np.ndarray, spacing: tuple) -> float:
    """
    Computes the Average Surface Distance (ASD) in millimeters.
    Returns np.nan if either array is completely empty.
    """
    asd_score = np.nan
    p_points = np.argwhere(p_array > 0)
    g_points = np.argwhere(g_array > 0)
    
    if len(p_points) > 0 and len(g_points) > 0:
        p_points_mm = p_points * spacing
        g_points_mm = g_points * spacing
        
        tree_p = cKDTree(p_points_mm)
        tree_g = cKDTree(g_points_mm)
        
        dist_p_to_g, _ = tree_p.query(g_points_mm, k=1)
        dist_g_to_p, _ = tree_g.query(p_points_mm, k=1)
        
        asd_score = (np.mean(dist_p_to_g) + np.mean(dist_g_to_p)) / 2.0
        
    return asd_score

def calculate_lesion_f1(p_array: np.ndarray, g_array: np.ndarray, overlap_threshold: float = 0.1) -> float:
    """
    Computes the Lesion-wise F1 Score based on connected components overlap.
    """
    f1_score = 0.0
    struct = np.ones((3, 3, 3)) if p_array.ndim == 3 else None
    
    gt_labeled, num_gt = label(g_array, structure=struct)
    pred_labeled, num_pred = label(p_array, structure=struct)

    if num_gt == 0 and num_pred == 0:
        f1_score = 1.0
    elif num_gt > 0 and num_pred > 0:
        tp = 0
        matched_pred_lesions = set()

        for i in range(1, num_gt + 1):
            gt_lesion_mask = (gt_labeled == i)
            gt_vol = np.sum(gt_lesion_mask)
            total_overlap = np.sum(gt_lesion_mask & (p_array > 0))
            
            if total_overlap / gt_vol >= overlap_threshold:
                tp += 1
                overlapping_preds = np.unique(pred_labeled[gt_lesion_mask])
                for pred_id in overlapping_preds:
                    if pred_id > 0:
                        matched_pred_lesions.add(pred_id)

        fn = num_gt - tp
        fp = num_pred - len(matched_pred_lesions)

        if tp > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 2 * (precision * recall) / (precision + recall)
            
    return f1_score

def post_process_mask(mask_array: np.ndarray, spacing: tuple, min_vol_ml: float = 0.025) -> np.ndarray:
    """
    Removes small connected components (noise) and fills holes in the prediction mask.
    """
    min_vol_mm3 = min_vol_ml * 1000.0
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    min_voxels = int(np.ceil(min_vol_mm3 / voxel_vol_mm3))
    
    mask_bool = mask_array > 0
    mask_bool = remove_small_objects(mask_bool, min_size=min_voxels)
    
    spacing_np = np.array([spacing[2], spacing[1], spacing[0]])
    thick_axis = np.argmax(spacing_np)
    
    filled_mask = np.zeros_like(mask_bool)
    
    if np.max(spacing_np) / np.min(spacing_np) >= 2.0:
        if thick_axis == 0:
            for i in range(mask_bool.shape[0]):
                filled_mask[i] = binary_fill_holes(mask_bool[i])
        elif thick_axis == 1:
            for i in range(mask_bool.shape[1]):
                filled_mask[:, i] = binary_fill_holes(mask_bool[:, i])
        else:
            for i in range(mask_bool.shape[2]):
                filled_mask[:, :, i] = binary_fill_holes(mask_bool[:, :, i])
    else:
        filled_mask = binary_fill_holes(mask_bool)
        
    return filled_mask.astype(mask_array.dtype)