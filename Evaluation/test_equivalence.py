"""
Module: test_equivalence.py
Description: Performs Two One-Sided Tests (TOST) to determine statistical 
             equivalence between full precision (FP32) and quantized (FP16) models.
"""

import os
import datetime
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import SimpleITK as sitk
from tqdm import tqdm

from Evaluation.metrics import calculate_dice

def perform_tost(diffs: np.ndarray, margin: float, alpha: float = 0.05) -> dict:
    """
    Executes a paired TOST for equivalence on a given array of differences.
    Returns a dictionary containing the confidence intervals and equivalence verdict.
    """
    result = {
        "mean_diff": 0.0,
        "ci_lower": 0.0,
        "ci_upper": 0.0,
        "tost_p_val": 1.0,
        "is_equivalent": "No"
    }
    
    n = len(diffs)
    if n > 0:
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1) if n > 1 else 0.0
        
        result["mean_diff"] = mean_diff
        
        if n > 1 and std_diff > 0:
            se = std_diff / np.sqrt(n)
            
            t1 = (mean_diff - (-margin)) / se
            p1 = stats.t.sf(t1, n - 1)
            
            t2 = (mean_diff - margin) / se
            p2 = stats.t.cdf(t2, n - 1)
            
            tost_p_val = max(p1, p2)
            
            moe = stats.t.ppf(1 - alpha, n - 1) * se 
            ci_lower = mean_diff - moe
            ci_upper = mean_diff + moe
            
            result["ci_lower"] = ci_lower
            result["ci_upper"] = ci_upper
            result["tost_p_val"] = tost_p_val
            result["is_equivalent"] = "Yes" if (tost_p_val < alpha) else "No"
        else:
            result["ci_lower"] = mean_diff
            result["ci_upper"] = mean_diff
            result["tost_p_val"] = 0.0 if abs(mean_diff) < margin else 1.0
            result["is_equivalent"] = "Yes" if abs(mean_diff) < margin else "No"
            
    return result

def test_model_equivalence(gt_dir: str, dir_fp32: str, dir_fp16: str, model_name: str) -> dict:
    """
    Evaluates the predictions of two precision variants of a model against 
    ground truth and computes their statistical equivalence.
    """
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.nii.gz')]
    model_fp32_dices = []
    model_fp16_dices = []
    mask_agreements = []
    
    for gt_name in tqdm(gt_files, desc=f"Analyzing {model_name} Equivalence"):
        sub_id = gt_name.replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, gt_name)
        
        path_fp32 = os.path.join(dir_fp32, f"{sub_id}.nii.gz")
        path_fp16 = os.path.join(dir_fp16, f"{sub_id}.nii.gz")
        
        if os.path.exists(path_fp32) and os.path.exists(path_fp16):
            gt_img = sitk.ReadImage(gt_path)
            gt_array = sitk.GetArrayFromImage(gt_img)
            
            arr_fp32 = sitk.GetArrayFromImage(sitk.ReadImage(path_fp32))
            arr_fp16 = sitk.GetArrayFromImage(sitk.ReadImage(path_fp16))
            
            dice_fp32 = calculate_dice(arr_fp32, gt_array)
            dice_fp16 = calculate_dice(arr_fp16, gt_array)
            agreement = calculate_dice(arr_fp32, arr_fp16)
            
            model_fp32_dices.append(dice_fp32)
            model_fp16_dices.append(dice_fp16)
            mask_agreements.append(agreement)
            
    final_stats = {
        "Model": model_name,
        "Mean_Dice_FP32": 0.0,
        "Mean_Dice_FP16": 0.0,
        "Mean_Mask_Agreement_Dice": 0.0,
        "Mean_Difference": 0.0,
        "90%_CI_Lower": 0.0,
        "90%_CI_Upper": 0.0,
        "TOST_P_Value": 1.0,
        "Equivalent_at_5%_Alpha?": "No"
    }
    
    if len(model_fp32_dices) > 0:
        diffs = np.array(model_fp32_dices) - np.array(model_fp16_dices)
        tost_results = perform_tost(diffs, margin=1e-3, alpha=0.05)
        
        final_stats["Mean_Dice_FP32"] = np.mean(model_fp32_dices)
        final_stats["Mean_Dice_FP16"] = np.mean(model_fp16_dices)
        final_stats["Mean_Mask_Agreement_Dice"] = np.mean(mask_agreements)
        final_stats["Mean_Difference"] = tost_results["mean_diff"]
        final_stats["90%_CI_Lower"] = tost_results["ci_lower"]
        final_stats["90%_CI_Upper"] = tost_results["ci_upper"]
        final_stats["TOST_P_Value"] = tost_results["tost_p_val"]
        final_stats["Equivalent_at_5%_Alpha?"] = tost_results["is_equivalent"]
        
    return final_stats

def run_equivalence_pipeline(models: list, gt_dir: str, preds_base_dir: str, output_dir: str) -> None:
    """
    Iterates over a list of models, computes their TOST equivalence, 
    and exports the aggregated results to a CSV file.
    """
    equivalence_results = []
    
    for model_name in models:
        dir_fp32 = os.path.join(preds_base_dir, f"{model_name.lower()}_fp32")
        dir_fp16 = os.path.join(preds_base_dir, f"{model_name.lower()}_fp16")
        
        stats_dict = test_model_equivalence(gt_dir, dir_fp32, dir_fp16, model_name)
        if stats_dict["Mean_Dice_FP32"] > 0: 
            equivalence_results.append(stats_dict)
            
    if equivalence_results:
        os.makedirs(output_dir, exist_ok=True)
        df_eq = pd.DataFrame(equivalence_results)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"FP32_vs_FP16_Equivalence_TOST_{timestamp}.csv")
        df_eq.to_csv(csv_path, index=False)
        print(f"\nEquivalence Results saved to: {csv_path}")
        
    return None

def main() -> None:
    """
    Main entry point handling command-line arguments for TOST testing.
    """
    parser = argparse.ArgumentParser(description="Perform TOST equivalence testing between FP32 and FP16 outputs.")
    parser.add_argument("--gt-dir", type=str, required=True, help="Directory containing Ground Truth masks.")
    parser.add_argument("--preds-base-dir", type=str, default="preds", help="Base directory containing prediction model subfolders.")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory where output CSVs will be saved.")
    parser.add_argument("--models", type=str, nargs='+', default=[
        "Teacher", "Femto", "Pico", "Nano", 
        "ExtraExtraLight", "ExtraLight", "Light", 
        "Small", "Medium", "Large"
    ], help="List of model names to evaluate.")
    
    args = parser.parse_args()
    
    run_equivalence_pipeline(
        models=args.models,
        gt_dir=args.gt_dir,
        preds_base_dir=args.preds_base_dir,
        output_dir=args.output_dir
    )
    return None

if __name__ == "__main__":
    main()