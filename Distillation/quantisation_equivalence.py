import os
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.stats as stats
from scipy.ndimage import label
from scipy.spatial import cKDTree
from tqdm import tqdm
import datetime

# 1. Environment Setup
os.environ['nnUNet_raw'] = "/home/ymahe/Desktop/StrokeSegLight/Distillation/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "/home/ymahe/Desktop/StrokeSegLight/Distillation/nnUNet_preprocessed"
os.environ['nnUNet_results'] = "/home/ymahe/Desktop/StrokeSegLight/Distillation/nnUNet_results"

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import load_json, join

# --- IMPORT ALL STUDENT MODELS ---
from models import *

# --- UTILITY: PARAMETER COUNTER ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- UTILITY: DICE CALCULATION ---
def calculate_dice(p_array, g_array):
    if np.sum(p_array) + np.sum(g_array) == 0: return 1.0
    return 2.0 * np.sum(p_array * g_array) / (np.sum(p_array) + np.sum(g_array))

# --- UTILITY: AVERAGE SURFACE DISTANCE CALCULATION ---
def calculate_asd(p_array, g_array, spacing):
    p_points = np.argwhere(p_array > 0)
    g_points = np.argwhere(g_array > 0)
    
    if len(p_points) == 0 or len(g_points) == 0:
        return np.nan
    
    p_points_mm = p_points * spacing
    g_points_mm = g_points * spacing
    
    tree_p = cKDTree(p_points_mm)
    tree_g = cKDTree(g_points_mm)
    
    dist_p_to_g, _ = tree_p.query(g_points_mm, k=1)
    dist_g_to_p, _ = tree_g.query(p_points_mm, k=1)
    
    asd = (np.mean(dist_p_to_g) + np.mean(dist_g_to_p)) / 2.0
    return asd

# --- UTILITY: SUBJECT-WISE DETECTION METRICS ---
def calculate_subject_detection_metrics(p_array, g_array):
    global_dice = calculate_dice(p_array, g_array)
    """
    struct = np.ones((3, 3, 3)) if p_array.ndim == 3 else None
    gt_labeled, num_gt = label(g_array, structure=struct)
    pred_labeled, num_pred = label(p_array, structure=struct)

    gt_found_count = 0
    for i in range(1, num_gt + 1):
        gt_cc_mask = (gt_labeled == i)
        gt_vol = np.sum(gt_cc_mask)
        overlap = np.sum(gt_cc_mask & (p_array > 0))
        if overlap / gt_vol >= 0.10:
            gt_found_count += 1

    tp_binary = 1 if (num_gt > 0 and gt_found_count > 0) else 0
    fn_binary = 1 if (num_gt > 0 and gt_found_count == 0) else 0

    fp_count = 0
    filtered_p_array = np.copy(p_array)
    for i in range(1, num_pred + 1):
        pred_cc_mask = (pred_labeled == i)
        pred_vol = np.sum(pred_cc_mask)
        overlap = np.sum(pred_cc_mask & (g_array > 0))
        
        if overlap / pred_vol < 0.10:
            if pred_vol > 25:
                fp_count += 1
                filtered_p_array[pred_cc_mask] = 0
            else:
                filtered_p_array[pred_cc_mask] = 0

    lesion_dice = calculate_dice(filtered_p_array, g_array)
    asd = calculate_asd(filtered_p_array, g_array, 1) if np.sum(p_array) > 0 and np.sum(g_array) > 0 else np.nan
    if lesion_dice == 0.0:
        lesion_dice = np.nan
    """
    return 0, 0, 0, global_dice, 0, np.nan


# --- MAIN EVALUATION FUNCTION ---
def run_evaluation(model_name: str, precision: str = "fp32"):
    print(f"\n=============================================")
    print(f"   EVALUATING MODEL: {model_name.upper()} | {precision.upper()}")
    print(f"=============================================")
    
    test_img_dir = "Test_Set_ATLAS_2.1/images"
    test_gt_dir = "Test_Set_ATLAS_2.1/masks"
    
    pred_dir = f"preds/{model_name.lower()}_{precision}"
    
    if model_name.lower() == "teacher":
        model_folder = join(os.environ['nnUNet_results'], "Dataset999", "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres")
    else:
        get_student_fn_map = {
            "Femto": ("nnUNetTrainer_KD_Femto", get_femto_student),
            "Pico": ("nnUNetTrainer_KD_Pico", get_pico_student),
            "Nano": ("nnUNetTrainer_KD_Nano", get_nano_student),
            "Light": ("nnUNetTrainer_KD_Light", get_light_student),
            "Small": ("nnUNetTrainer_KD_Small", get_small_student),
            "Medium": ("nnUNetTrainer_KD_Medium", get_medium_student),
            "Large": ("nnUNetTrainer_KD_Large", get_large_student),
            "ExtraLight": ("nnUNetTrainer_KD_ExtraLight", get_extra_light_student),
            "ExtraExtraLight": ("nnUNetTrainer_KD_ExtraExtraLight", get_extra_extralight_student)
        }
        trainer_name, get_student_fn = get_student_fn_map[model_name]
        model_folder = join(os.environ['nnUNet_results'], "Dataset999", f"{trainer_name}__nnUNetResEncUNetMPlans__3d_fullres")

    os.makedirs(pred_dir, exist_ok=True)

    test_images = [f for f in os.listdir(test_img_dir) if f.endswith('_0000.nii.gz') or f.endswith('.nii.gz')]
    existing_preds = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]
    
    if len(existing_preds) < len(test_images):
        print(f"Predictions missing in '{pred_dir}'. Initializing inference...")
        
        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
            perform_everything_on_device=True, device=torch.device('cuda'), verbose=False, allow_tqdm=True
        )
        
        if not os.path.exists(model_folder):
            raise FileNotFoundError(f"Could not find model folder: {model_folder}")

        if model_name.lower() == "teacher":
            predictor.initialize_from_trained_model_folder(model_folder, use_folds=(2,), checkpoint_name='checkpoint_best.pth')
        else:
            plans = load_json(join(model_folder, 'plans.json'))
            dataset_json = load_json(join(model_folder, 'dataset.json'))
            plans_manager = PlansManager(plans)
            configuration_manager = plans_manager.get_configuration('3d_fullres')

            student_network = get_student_fn().cuda()
            
            checkpoint_path = join(model_folder, "fold_2", "checkpoint_best.pth")
            checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
            student_network.load_state_dict(checkpoint['network_weights'])
            student_network.eval()

            mirror_axes = checkpoint.get('inference_allowed_mirroring_axes', None)
            
            predictor.manual_initialization(
                network=student_network, plans_manager=plans_manager, configuration_manager=configuration_manager,
                parameters=[checkpoint['network_weights']], dataset_json=dataset_json, trainer_name=trainer_name,
                inference_allowed_mirroring_axes=mirror_axes
            )
        
        dtype = torch.float16 if precision == "fp16" else torch.float32
        predictor.network.to(dtype=dtype)
        
        print(f"Running Prediction for {model_name} in {precision}...")
        with torch.autocast(device_type='cuda', dtype=dtype):
            predictor.predict_from_files(test_img_dir, pred_dir, save_probabilities=False, overwrite=True,
                                         num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                         folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    else:
        print(f"Predictions already exist in '{pred_dir}'. Skipping inference.")

    # Calculate Individual Model Metrics
    print(f"\n--- Calculating Subject-Wise Metrics for {model_name} ({precision}) ---")
    results = []
    gt_files = [f for f in os.listdir(test_gt_dir) if f.endswith('.nii.gz')]
    
    for gt_name in tqdm(gt_files, desc="Evaluating Subjects"):
        sub_id = gt_name.replace(".nii.gz", "")
        gt_path = os.path.join(test_gt_dir, gt_name)
        gt_img = sitk.ReadImage(gt_path)
        gt_array = sitk.GetArrayFromImage(gt_img)
        spacing = gt_img.GetSpacing()
        
        voxel_volume_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
        gt_volume_ml = np.sum(gt_array > 0) * voxel_volume_ml
        
        pred_path = os.path.join(pred_dir, f"{sub_id}.nii.gz")
        
        if os.path.exists(pred_path):
            p_array = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
            tp, fp, fn, dice, l_dice, asd = calculate_subject_detection_metrics(p_array, gt_array)
        else:
            tp, fp, fn, dice, l_dice, asd= 0, 0, (1 if np.sum(gt_array) > 0 else 0), 0.0, 0.0, np.nan
        
        results.append({
            "Subject": sub_id,
            "Model": model_name,
            "Precision": precision,
            "GT_Volume_ml": gt_volume_ml, 
            "TP_Subject": tp,   
            "FP_Count": fp,     
            "FN_Subject": fn,   
            "Global_Dice": dice,
            "Lesion_Dice": l_dice,
            "Lesion_F1" : (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
            "Average_Surface_Distance_mm": asd
        })

    df = pd.DataFrame(results)
    
    valid_lesions = df[df['GT_Volume_ml'] > 0]['GT_Volume_ml']
    
    if not valid_lesions.empty:
        q25 = valid_lesions.quantile(0.25)
        q75 = valid_lesions.quantile(0.75)
        
        def categorize_size(vol):
            if vol == 0: return 'None'
            elif vol <= q25: return 'S'
            elif vol <= q75: return 'M'
            else: return 'L'
                
        df['Size'] = df['GT_Volume_ml'].apply(categorize_size)
    else:
        df['Size'] = 'None'
    
    os.makedirs("evaluation", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"evaluation/Subject_Detection_{model_name}_{precision}_{timestamp}.csv"
    
    df.to_csv(csv_path, index=False)


# --- 95% EQUIVALENCE TESTING FUNCTION (TOST) ---
def run_equivalence_test(models):
    print("\n=============================================")
    print("   RUNNING TOST EQUIVALENCE TEST (FP32 vs FP16) ")
    print("=============================================")
    
    test_gt_dir = "Test_Set_ATLAS_2.1/masks"
    gt_files = [f for f in os.listdir(test_gt_dir) if f.endswith('.nii.gz')]
    
    equivalence_results = []
    
    for model_name in models:
        print(f"Analyzing {model_name}...")
        dir_fp32 = f"preds/{model_name.lower()}_fp32"
        dir_fp16 = f"preds/{model_name.lower()}_fp16"
        
        model_fp32_dices = []
        model_fp16_dices = []
        mask_agreements = []
        
        for gt_name in tqdm(gt_files, desc=f"Subjects ({model_name})"):
            sub_id = gt_name.replace(".nii.gz", "")
            gt_path = os.path.join(test_gt_dir, gt_name)
            
            gt_img = sitk.ReadImage(gt_path)
            gt_array = sitk.GetArrayFromImage(gt_img)
            
            path_fp32 = os.path.join(dir_fp32, f"{sub_id}.nii.gz")
            path_fp16 = os.path.join(dir_fp16, f"{sub_id}.nii.gz")
            
            if os.path.exists(path_fp32) and os.path.exists(path_fp16):
                arr_fp32 = sitk.GetArrayFromImage(sitk.ReadImage(path_fp32))
                arr_fp16 = sitk.GetArrayFromImage(sitk.ReadImage(path_fp16))
                
                # Compare against GT
                _, _, _, dice_fp32, _, _ = calculate_subject_detection_metrics(arr_fp32, gt_array)
                _, _, _, dice_fp16, _, _ = calculate_subject_detection_metrics(arr_fp16, gt_array)
                
                # Direct Agreement between FP32 and FP16 masks
                agreement = calculate_dice(arr_fp32, arr_fp16)
                
                model_fp32_dices.append(dice_fp32)
                model_fp16_dices.append(dice_fp16)
                mask_agreements.append(agreement)
                
        # Statistical Test (Paired TOST for Equivalence)
        if len(model_fp32_dices) > 0:
            margin = 1e-3  # Define equivalence margin
            alpha = 0.05   # Significance level
            
            diffs = np.array(model_fp32_dices) - np.array(model_fp16_dices)
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 0
            n = len(diffs)
            
            if n > 1 and std_diff > 0:
                se = std_diff / np.sqrt(n)
                
                # TOST Test 1: Mean difference > -margin
                t1 = (mean_diff - (-margin)) / se
                p1 = stats.t.sf(t1, n - 1)  # survival function (1 - CDF)
                
                # TOST Test 2: Mean difference < margin
                t2 = (mean_diff - margin) / se
                p2 = stats.t.cdf(t2, n - 1)
                
                tost_p_val = max(p1, p2)
                
                # For TOST with alpha=0.05, we report the 90% CI (not 95%) 
                moe = stats.t.ppf(1 - alpha, n-1) * se 
                ci_lower = mean_diff - moe
                ci_upper = mean_diff + moe
                
                is_equivalent = "Yes" if (tost_p_val < alpha) else "No"
            else:
                # Edge case: No variance (identical arrays) or n=1
                mean_diff = 0.0 if n == 0 else mean_diff
                ci_lower, ci_upper = mean_diff, mean_diff
                tost_p_val = 0.0 if abs(mean_diff) < margin else 1.0
                is_equivalent = "Yes" if abs(mean_diff) < margin else "No"
            
            equivalence_results.append({
                "Model": model_name,
                "Mean_Dice_FP32": np.mean(model_fp32_dices),
                "Mean_Dice_FP16": np.mean(model_fp16_dices),
                "Mean_Mask_Agreement_Dice": np.mean(mask_agreements),
                "Mean_Difference": mean_diff,
                "90%_CI_Lower": ci_lower,
                "90%_CI_Upper": ci_upper,
                "TOST_P_Value": tost_p_val,
                "Equivalent_at_5%_Alpha?": is_equivalent
            })
        
    # Export Equivalence CSV
    if equivalence_results:
        df_eq = pd.DataFrame(equivalence_results)
        os.makedirs("evaluation", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"evaluation/FP32_vs_FP16_Equivalence_TOST_{timestamp}.csv"
        df_eq.to_csv(csv_path, index=False)
        print(f"\nEquivalence Results saved to: {csv_path}")
        print(df_eq.to_string())
    else:
        print("\nNo equivalence results could be calculated (missing predictions).")

if __name__ == "__main__":
    models_to_test = [
        "Teacher", "Femto", "Pico", "Nano", 
        "ExtraExtraLight", "ExtraLight", "Light", 
        "Small", "Medium", "Large"
    ]
    
    # 1. Run all inferences and metric calculations in both precisions
    for model in models_to_test:
        try:
            run_evaluation(model, precision="fp32")
            run_evaluation(model, precision="fp16")
        except Exception as e:
            print(f"Skipping {model} due to error: {e}")
            
    # 2. Run the Equivalence Analysis
    run_equivalence_test(models_to_test)