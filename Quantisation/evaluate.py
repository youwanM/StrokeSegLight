import os
import argparse
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label, binary_fill_holes
from skimage.morphology import remove_small_objects
from tqdm import tqdm
import torch.nn as nn
import datetime

# 1. Environment Setup
os.environ['nnUNet_raw'] = "/path/to/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "/path/to/nnUNet_preprocessed"
os.environ['nnUNet_results'] = "/path/to/nnUNet_results"

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import load_json, join

# --- IMPORT ALL STUDENT MODELS ---
from models import get_light_student, get_small_student, get_medium_student

# --- UTILITY: DICE CALCULATION ---
def calculate_dice(p_array, g_array):
    if np.sum(p_array) + np.sum(g_array) == 0: return 1.0
    return 2.0 * np.sum(p_array * g_array) / (np.sum(p_array) + np.sum(g_array))

# --- UTILITY: LESION-WISE F1 CALCULATION ---
def calculate_lesion_f1(p_array, g_array, overlap_threshold=0.1):
    struct = np.ones((3, 3, 3)) if p_array.ndim == 3 else None
    gt_labeled, num_gt = label(g_array, structure=struct)
    pred_labeled, num_pred = label(p_array, structure=struct)

    if num_gt == 0 and num_pred == 0: return 1.0
    if num_gt == 0 or num_pred == 0: return 0.0

    tp = 0
    matched_pred_lesions = set()

    for i in range(1, num_gt + 1):
        gt_lesion_mask = (gt_labeled == i)
        gt_vol = np.sum(gt_lesion_mask)
        total_overlap = np.sum(gt_lesion_mask & (p_array > 0))
        
        if total_overlap / gt_vol >= overlap_threshold:
            tp += 1
            overlapping_preds = np.unique(pred_labeled[gt_lesion_mask])
            matched_pred_lesions.update(overlapping_preds[overlapping_preds > 0])

    fn = num_gt - tp
    fp = num_pred - len(matched_pred_lesions)

    if tp == 0: return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)

# --- UTILITY: POST-PROCESSING ---
def post_process_mask(mask_array, spacing, min_vol_ml=0.025):
    min_vol_mm3 = min_vol_ml * 1000.0
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    min_voxels = int(np.ceil(min_vol_mm3 / voxel_vol_mm3))
    
    mask_bool = mask_array > 0
    mask_bool = remove_small_objects(mask_bool, min_size=min_voxels)
    
    spacing_np = np.array([spacing[2], spacing[1], spacing[0]])
    
    if np.max(spacing_np) / np.min(spacing_np) >= 2.0:
        thick_axis = np.argmax(spacing_np)
        filled_mask = np.zeros_like(mask_bool)
        
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

# --- UTILITY: PARAMETER COUNTER ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_comparison(student_size: str, run_fp16: bool):
    print(f"=============================================")
    print(f"   EVALUATING STUDENT SIZE: {student_size.upper()}")
    print(f"=============================================")

    if student_size == "Light":
        get_student_fn = get_light_student
        trainer_name = "nnUNetTrainer_KD_Light"
    elif student_size == "Small":
        get_student_fn = get_small_student
        trainer_name = "nnUNetTrainer_KD_Small"
    elif student_size == "Medium":
        get_student_fn = get_medium_student
        trainer_name = "nnUNetTrainer_KD_Medium"
    else:
        raise ValueError("student_size must be 'Light', 'Small', or 'Medium'")

    test_img_dir = "Test_Set_ATLAS_2.1/images"
    test_gt_dir = "Test_Set_ATLAS_2.1/masks"
    
    # Configure required precision evaluations
    precisions_to_evaluate = ["FP32"]
    if run_fp16: precisions_to_evaluate.append("FP16")

    teacher_out_dir = "preds/teacher"
    student_out_dirs = {p: f"preds/student_{student_size.lower()}_{p.lower()}" for p in precisions_to_evaluate}
    
    os.makedirs(teacher_out_dir, exist_ok=True)
    for path in student_out_dirs.values():
        os.makedirs(path, exist_ok=True)

    teacher_model_folder = join(os.environ['nnUNet_results'], "Dataset999", "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres")
    student_model_folder = join(os.environ['nnUNet_results'], "Dataset999", f"{trainer_name}__nnUNetResEncUNetMPlans__3d_fullres")

    # ==========================================
    # 1. SETUP & RUN TEACHER PREDICTOR
    # ==========================================
    print("\n--- Setting up Teacher ---")
    teacher_pred = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=False, 
            perform_everything_on_device=True, device=torch.device('cuda'), verbose=False, allow_tqdm=True
        )
    teacher_pred.initialize_from_trained_model_folder(teacher_model_folder, use_folds=(2,), checkpoint_name='checkpoint_best.pth')
        
    teacher_params = count_parameters(teacher_pred.network)   
    print(f"Teacher Parameters: {teacher_params / 1e6:.2f} Million")
    
    if os.path.exists(teacher_out_dir) and len(os.listdir(teacher_out_dir)) > 0:
        print(f"Teacher predictions already exist in '{teacher_out_dir}'. Skipping Teacher prediction step.")
    else:
        print("Running Teacher Prediction...")
        teacher_pred.predict_from_files(test_img_dir, teacher_out_dir, save_probabilities=False, overwrite=True,
                                        num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                        folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # ==========================================
    # 2. SETUP & RUN STUDENT PREDICTORS
    # ==========================================
    if not os.path.exists(student_model_folder):
        raise FileNotFoundError(f"Could not find Student model folder: {student_model_folder}\nDid you train this size yet?")

    plans = load_json(join(student_model_folder, 'plans.json'))
    dataset_json = load_json(join(student_model_folder, 'dataset.json'))
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration('3d_fullres')
    
    checkpoint_path = join(student_model_folder, "fold_2", "checkpoint_best.pth")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    mirror_axes = checkpoint.get('inference_allowed_mirroring_axes', None)

    for precision in precisions_to_evaluate:
        print(f"\n--- Setting up {student_size} Student ({precision}) ---")
        
        device = torch.device('cuda')
        
        student_pred = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
            perform_everything_on_device=(), 
            device=device, verbose=False, allow_tqdm=True
        )
        
        student_network = get_student_fn()
        student_network.load_state_dict(checkpoint['network_weights'])
        
        # Apply Quantizations / Type Casting
        if precision == "FP32":
            student_network = student_network.cuda()
        elif precision == "FP16":
            student_network = student_network.half().cuda()
            
        student_network.eval()
        
        student_pred.manual_initialization(
            network=student_network, plans_manager=plans_manager, configuration_manager=configuration_manager,
            parameters=[student_network.state_dict()], dataset_json=dataset_json, trainer_name=trainer_name,
            inference_allowed_mirroring_axes=mirror_axes
        )
        
        print(f"Running Student Prediction ({precision})...")
        student_pred.predict_from_files(test_img_dir, student_out_dirs[precision], save_probabilities=False, overwrite=True,
                                        num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                        folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # ==========================================
    # 3. CALCULATE METRICS & EXPORT
    # ==========================================
    print("\n--- Calculating Metrics (Dice, Lesion F1, Size) with Post-Processing ---")
    results = []
    gt_files = [f for f in os.listdir(test_gt_dir) if f.endswith('.nii.gz')]
    
    for gt_name in tqdm(gt_files, desc="Evaluating Subjects"):
        sub_id = gt_name.replace(".nii.gz", "")
        gt_path = os.path.join(test_gt_dir, gt_name)
        
        gt_img = sitk.ReadImage(gt_path)
        gt_array = sitk.GetArrayFromImage(gt_img)
        spacing = gt_img.GetSpacing() 
        gt_size_voxels = np.sum(gt_array) 
        
        t_pred_path = os.path.join(teacher_out_dir, f"{sub_id}.nii.gz")
        
        # Teacher Metrics
        if os.path.exists(t_pred_path):
            t_array = sitk.GetArrayFromImage(sitk.ReadImage(t_pred_path))
            t_array = post_process_mask(t_array, spacing)
            t_dice = calculate_dice(t_array, gt_array)
            t_f1 = calculate_lesion_f1(t_array, gt_array, overlap_threshold=0.1)
        else:
            t_dice, t_f1 = 0.0, 0.0
            
        row_data = {
            "Subject": sub_id,
            "GT_Size_Voxels": gt_size_voxels,
            "Teacher_Dice": t_dice,
            "Teacher_Lesion_F1": t_f1
        }
        
        # Student Metrics (Loop through available precisions)
        for precision, s_dir in student_out_dirs.items():
            s_pred_path = os.path.join(s_dir, f"{sub_id}.nii.gz")
            if os.path.exists(s_pred_path):
                s_array = sitk.GetArrayFromImage(sitk.ReadImage(s_pred_path))
                s_array = post_process_mask(s_array, spacing)
                s_dice = calculate_dice(s_array, gt_array)
                s_f1 = calculate_lesion_f1(s_array, gt_array, overlap_threshold=0.1)
            else:
                s_dice, s_f1 = 0.0, 0.0
                
            row_data[f"Student_{precision}_Dice"] = s_dice
            row_data[f"{precision}_Dice_Difference"] = s_dice - t_dice
            row_data[f"Student_{precision}_Lesion_F1"] = s_f1
            row_data[f"{precision}_F1_Difference"] = s_f1 - t_f1
            
        results.append(row_data)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate Percentiles and Stratify
    p25 = df['GT_Size_Voxels'].quantile(0.25)
    p75 = df['GT_Size_Voxels'].quantile(0.75)
    
    conditions = [
        df['GT_Size_Voxels'] <= p25,
        (df['GT_Size_Voxels'] > p25) & (df['GT_Size_Voxels'] <= p75),
        df['GT_Size_Voxels'] > p75
    ]
    choices = ['S', 'M', 'L']
    df['Mask_Size_Category'] = np.select(conditions, choices, default='Unknown')

    print(f"\nSize Thresholds - 25th Percentile: {p25:.1f} voxels | 75th Percentile: {p75:.1f} voxels")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"KD_Experiment_Results_{student_size}_{timestamp}.csv"
    result_folder = "evaluation"
    os.makedirs(result_folder, exist_ok=True)
    
    # Reorder columns dynamically based on requested precisions
    cols = ["Subject", "GT_Size_Voxels", "Mask_Size_Category", "Teacher_Dice", "Teacher_Lesion_F1"]
    for p in precisions_to_evaluate:
        cols.extend([f"Student_{p}_Dice", f"{p}_Dice_Difference", f"Student_{p}_Lesion_F1", f"{p}_F1_Difference"])
    
    df = df[cols]
    
    df.to_csv(os.path.join(result_folder, csv_filename), index=False)
    print(f"Results saved to: {os.path.join(result_folder, csv_filename)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Student Models with Optional Quantization")
    parser.add_argument("--fp16", action="store_true", help="Run FP16 evaluation alongside FP32")
    args = parser.parse_args()

    for student_size in ["ExtraExtraLight", "ExtraLight", "Light", "Small", "Medium", "Large"]:
        run_comparison(student_size=student_size, run_fp16=args.fp16)
