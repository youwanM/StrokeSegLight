"""
Module: evaluate_models.py
Description: Evaluates model predictions against Ground Truth using standard 
             metrics (Dice, F1, ASD) and stratifies results by lesion volume.
"""

import os
import datetime
import argparse
import pandas as pd
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

from Evaluation.metrics import (
    calculate_dice, 
    calculate_asd, 
    calculate_lesion_f1, 
    post_process_mask
)

def evaluate_subject_predictions(pred_dir: str, gt_dir: str, model_name: str, precision: str = "fp32") -> pd.DataFrame:
    """
    Evaluates all subjects in the ground truth directory against predictions.
    Returns a DataFrame containing the computed metrics.
    """
    results = []
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.nii.gz')]
    
    for gt_name in tqdm(gt_files, desc=f"Evaluating {model_name} ({precision})"):
        sub_id = gt_name.replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, gt_name)
        
        gt_img = sitk.ReadImage(gt_path)
        gt_array = sitk.GetArrayFromImage(gt_img)
        spacing = gt_img.GetSpacing()
        
        voxel_volume_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
        gt_volume_ml = np.sum(gt_array > 0) * voxel_volume_ml
        
        pred_path = os.path.join(pred_dir, f"{sub_id}.nii.gz")
        
        # SESE default initialization
        dice = 0.0
        l_f1 = 0.0
        asd = np.nan
        
        if os.path.exists(pred_path):
            p_img = sitk.ReadImage(pred_path)
            p_array = sitk.GetArrayFromImage(p_img)
            p_array = post_process_mask(p_array, spacing)
            
            dice = calculate_dice(p_array, gt_array)
            l_f1 = calculate_lesion_f1(p_array, gt_array)
            asd = calculate_asd(p_array, gt_array, spacing)
            
        results.append({
            "Subject": sub_id,
            "Model": model_name,
            "Precision": precision,
            "GT_Volume_ml": gt_volume_ml,
            "Global_Dice": dice,
            "Lesion_F1": l_f1,
            "Average_Surface_Distance_mm": asd
        })
        
    df_results = pd.DataFrame(results)
    return df_results

def categorize_lesion_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratifies lesions into Small (S), Medium (M), and Large (L) categories 
    based on the 25th and 75th percentiles of valid lesion volumes.
    """
    df_stratified = df.copy()
    valid_lesions = df_stratified[df_stratified['GT_Volume_ml'] > 0]['GT_Volume_ml']
    
    if not valid_lesions.empty:
        q25 = valid_lesions.quantile(0.25)
        q75 = valid_lesions.quantile(0.75)
        
        conditions = [
            df_stratified['GT_Volume_ml'] == 0,
            df_stratified['GT_Volume_ml'] <= q25,
            (df_stratified['GT_Volume_ml'] > q25) & (df_stratified['GT_Volume_ml'] <= q75),
            df_stratified['GT_Volume_ml'] > q75
        ]
        choices = ['None', 'S', 'M', 'L']
        df_stratified['Size'] = np.select(conditions, choices, default='Unknown')
    else:
        df_stratified['Size'] = 'None'
        
    return df_stratified

def run_evaluation_pipeline(pred_dir: str, gt_dir: str, output_dir: str, model_name: str) -> None:
    """
    Main execution pipeline to evaluate a model, stratify results, and save to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df_raw = evaluate_subject_predictions(pred_dir, gt_dir, model_name)
    df_final = categorize_lesion_size(df_raw)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"Evaluation_{model_name}_{timestamp}.csv")
    
    df_final.to_csv(csv_path, index=False)
    print(f"Results successfully saved to: {csv_path}")
    
    return None

def main() -> None:
    """
    Main entry point handling command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions against Ground Truth.")
    parser.add_argument("--pred-dir", type=str, required=True, help="Directory containing prediction NIfTI files.")
    parser.add_argument("--gt-dir", type=str, required=True, help="Directory containing Ground Truth NIfTI masks.")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory where output CSVs will be stored.")
    parser.add_argument("--model-name", type=str, default="Teacher", help="Name of the model being evaluated.")
    
    args = parser.parse_args()
    
    run_evaluation_pipeline(
        pred_dir=args.pred_dir, 
        gt_dir=args.gt_dir, 
        output_dir=args.output_dir, 
        model_name=args.model_name
    )
    return None

if __name__ == "__main__":
    main()