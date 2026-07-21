import os
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label, binary_fill_holes
from scipy.spatial import cKDTree
from skimage.morphology import remove_small_objects
from tqdm import tqdm
import datetime

# 1. Environment Setup
os.environ['nnUNet_raw'] = "/path/to/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "/path/to/nnUNet_preprocessed"
os.environ['nnUNet_results'] = "/path/to/nnUNet_results"

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
    """
    Computes binary TP/FN for the subject and counts FPs.
    Filters FPs to calculate Lesion Dice.
    """
    global_dice = calculate_dice(p_array, g_array)
    
    struct = np.ones((3, 3, 3)) if p_array.ndim == 3 else None
    gt_labeled, num_gt = label(g_array, structure=struct)
    pred_labeled, num_pred = label(p_array, structure=struct)

    # Track which GT lesions are "found"
    gt_found_count = 0
    for i in range(1, num_gt + 1):
        gt_cc_mask = (gt_labeled == i)
        gt_vol = np.sum(gt_cc_mask)
        overlap = np.sum(gt_cc_mask & (p_array > 0))
        if overlap / gt_vol >= 0.10:
            gt_found_count += 1

    # Subject-level TP/FN (Binary)
    tp_binary = 1 if (num_gt > 0 and gt_found_count > 0) else 0
    fn_binary = 1 if (num_gt > 0 and gt_found_count == 0) else 0

    # FP Logic: Count components > 25 voxels not hitting a GT lesion
    fp_count = 0
    filtered_p_array = np.copy(p_array)
    for i in range(1, num_pred + 1):
        pred_cc_mask = (pred_labeled == i)
        pred_vol = np.sum(pred_cc_mask)
        overlap = np.sum(pred_cc_mask & (g_array > 0))
        
        # If it doesn't meet the 10% overlap criteria on the GT...
        if overlap / pred_vol < 0.10:
            if pred_vol > 25:
                fp_count += 1
                filtered_p_array[pred_cc_mask] = 0
            else:
                # Optional: also remove tiny noise even if not counted as FP
                filtered_p_array[pred_cc_mask] = 0

    # Lesion Dice: Dice after clearing out the FPs
    lesion_dice = calculate_dice(filtered_p_array, g_array)
    asd = calculate_asd(filtered_p_array, g_array, 1) if np.sum(p_array) > 0 and np.sum(g_array) > 0 else np.nan
    # If Lesion Dice is 0 changing to NaN to avoid skewing averages when no lesions are detected
    if lesion_dice == 0.0:
        lesion_dice = np.nan

    return tp_binary, fp_count, fn_binary, global_dice, lesion_dice, asd

# --- UTILITY: POST-PROCESSING ---
def post_process_mask(mask_array, spacing, min_vol_ml=0.025):
    min_vol_mm3 = min_vol_ml * 1000.0
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    min_voxels = int(np.ceil(min_vol_mm3 / voxel_vol_mm3))
    
    mask_bool = mask_array > 0
    mask_bool = remove_small_objects(mask_bool, max_size=min_voxels - 1)
    
    spacing_np = np.array([spacing[2], spacing[1], spacing[0]])
    if np.max(spacing_np) / np.min(spacing_np) >= 2.0:
        thick_axis = np.argmax(spacing_np)
        filled_mask = np.zeros_like(mask_bool)
        for i in range(mask_bool.shape[thick_axis]):
            slices = [slice(None)] * 3
            slices[thick_axis] = i
            filled_mask[tuple(slices)] = binary_fill_holes(mask_bool[tuple(slices)])
    else:
        filled_mask = binary_fill_holes(mask_bool)
        
    return filled_mask.astype(mask_array.dtype)

# --- MAIN EVALUATION FUNCTION ---
# --- MAIN EVALUATION FUNCTION ---
def run_evaluation(model_name: str):
    print(f"\n=============================================")
    print(f"   EVALUATING MODEL: {model_name.upper()}")
    print(f"=============================================")
    
    test_img_dir = "Test_Set_ATLAS_2.1/images"
    test_gt_dir = "Test_Set_ATLAS_2.1/masks"
    
    # 1. Path Setup Based on Model
    if model_name.lower() == "teacher":
        pred_dir = "preds/teacher"
        model_folder = join(os.environ['nnUNet_results'], "Dataset999", "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres")
    else:
        pred_dir = f"preds/student_{model_name.lower()}"
        if model_name == "Femto":
            get_student_fn = get_femto_student
            trainer_name = "nnUNetTrainer_KD_Femto"
        elif model_name == "Pico":
            get_student_fn = get_pico_student
            trainer_name = "nnUNetTrainer_KD_Pico"
        elif model_name == "Nano":
            get_student_fn = get_nano_student
            trainer_name = "nnUNetTrainer_KD_Nano"
        elif model_name == "Light":
            get_student_fn = get_light_student
            trainer_name = "nnUNetTrainer_KD_Light"
        elif model_name == "Small":
            get_student_fn = get_small_student
            trainer_name = "nnUNetTrainer_KD_Small"
        elif model_name == "Medium":
            get_student_fn = get_medium_student
            trainer_name = "nnUNetTrainer_KD_Medium"
        elif model_name == "Large":
            get_student_fn = get_large_student
            trainer_name = "nnUNetTrainer_KD_Large"
        elif model_name == "ExtraLight":
            get_student_fn = get_extra_light_student
            trainer_name = "nnUNetTrainer_KD_ExtraLight"
        elif model_name == "ExtraExtraLight":
            get_student_fn = get_extra_extralight_student
            trainer_name = "nnUNetTrainer_KD_ExtraExtraLight"
        else:
            raise ValueError(f"Unknown model name: {model_name}")
            
        model_folder = join(os.environ['nnUNet_results'], "Dataset999", f"{trainer_name}__nnUNetResEncUNetMPlans__3d_fullres")

    os.makedirs(pred_dir, exist_ok=True)

    # 2. Check for existing predictions and run Inference if needed
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
            params = count_parameters(predictor.network)   
            print(f"Teacher Parameters: {params / 1e6:.2f} Million")
        else:
            plans = load_json(join(model_folder, 'plans.json'))
            dataset_json = load_json(join(model_folder, 'dataset.json'))
            plans_manager = PlansManager(plans)
            configuration_manager = plans_manager.get_configuration('3d_fullres')

            student_network = get_student_fn().cuda()
            params = count_parameters(student_network)
            print(f"Student Parameters: {params / 1e6:.2f} Million")

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
            
        print(f"Running Prediction for {model_name}...")
        predictor.predict_from_files(test_img_dir, pred_dir, save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    else:
        print(f"Predictions already exist in '{pred_dir}'. Skipping inference.")

    # 3. Calculate Metrics
    print(f"\n--- Calculating Subject-Wise Metrics for {model_name} ---")
    results = []
    gt_files = [f for f in os.listdir(test_gt_dir) if f.endswith('.nii.gz')]
    
    for gt_name in tqdm(gt_files, desc="Evaluating Subjects"):
        sub_id = gt_name.replace(".nii.gz", "")
        gt_path = os.path.join(test_gt_dir, gt_name)
        gt_img = sitk.ReadImage(gt_path)
        gt_array = sitk.GetArrayFromImage(gt_img)
        spacing = gt_img.GetSpacing()
        
        # --- NEW: Calculate Ground Truth Volume in mL ---
        voxel_volume_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
        gt_volume_ml = np.sum(gt_array > 0) * voxel_volume_ml
        
        pred_path = os.path.join(pred_dir, f"{sub_id}.nii.gz")
        
        if os.path.exists(pred_path):
            p_array = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
            p_array = post_process_mask(p_array, spacing)
            tp, fp, fn, dice, l_dice, asd = calculate_subject_detection_metrics(p_array, gt_array)
        else:
            tp, fp, fn, dice, l_dice, asd= 0, 0, (1 if np.sum(gt_array) > 0 else 0), 0.0, 0.0, np.nan
        
        results.append({
            "Subject": sub_id,
            "Model": model_name,
            "GT_Volume_ml": gt_volume_ml, # Append volume to results
            "TP_Subject": tp,   
            "FP_Count": fp,     
            "FN_Subject": fn,   
            "Global_Dice": dice,
            "Lesion_Dice": l_dice ,
            "Lesion_F1" : (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
            "Average_Surface_Distance_mm": asd
        })

    # 4. Data Processing and Export to CSV
    df = pd.DataFrame(results)
    
    # --- NEW: Stratify Lesion Size based on 25th and 75th quartiles ---
    # We filter out volumes of 0 so healthy controls don't skew the quartiles
    valid_lesions = df[df['GT_Volume_ml'] > 0]['GT_Volume_ml']
    
    if not valid_lesions.empty:
        q25 = valid_lesions.quantile(0.25)
        q75 = valid_lesions.quantile(0.75)
        
        def categorize_size(vol):
            if vol == 0:
                return 'None' # Or adjust if you want healthy subjects labeled differently
            elif vol <= q25:
                return 'S'
            elif vol <= q75:
                return 'M'
            else:
                return 'L'
                
        df['Size'] = df['GT_Volume_ml'].apply(categorize_size)
    else:
        # Fallback if no lesions exist in the dataset
        df['Size'] = 'None'
    
    os.makedirs("evaluation", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"evaluation/Subject_Detection_{model_name}_{timestamp}.csv"
    
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    if not valid_lesions.empty:
        print(f"Size Thresholds calculated -> S: <= {q25:.2f}mL, M: > {q25:.2f}mL & <= {q75:.2f}mL, L: > {q75:.2f}mL")

if __name__ == "__main__":
    # Evaluate Teacher
    run_evaluation("Teacher")
    
    # Evaluate Students
    run_evaluation("Femto")
    run_evaluation("Pico")
    run_evaluation("Nano")
    run_evaluation("ExtraExtraLight")
    run_evaluation("ExtraLight")
    run_evaluation("Light")
    run_evaluation("Small")
    run_evaluation("Medium")
    run_evaluation("Large")
