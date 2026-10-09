"""
Module: prepare_data.py
Description: Aligns, binarizes, and formats the ATLAS dataset for nnU-Net 
             training. Generates the dataset.json and launches preprocessing.
"""

import os
import json
import shutil
import argparse
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn.image import resample_to_img

def prepare_training_data(base_dir_path: str, dataset_id: int, nnunet_raw_path: str) -> None:
    """
    Processes the raw ATLAS data by resampling masks to match T1 images, 
    binarizes them, and structures the output for nnU-Net.
    """
    base_dir = Path(base_dir_path)
    images_dir = base_dir / "preprocessed"
    labels_dir = base_dir / "derivatives"

    nnunet_dataset_dir = Path(nnunet_raw_path) / f"Dataset{dataset_id}"
    train_images_dir = nnunet_dataset_dir / "imagesTr"
    train_labels_dir = nnunet_dataset_dir / "labelsTr"

    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    if images_dir.exists():
        subjects = [d for d in images_dir.iterdir() if d.is_dir()]
        print("Aligning masks to images and binarizing...")
        
        for sub in subjects:
            sub_id = sub.name
            src_img_path = images_dir / sub_id / "anat" / f"{sub_id}_T1w.nii.gz"
            src_seg_path = labels_dir / sub_id / "seg" / f"{sub_id}_seg.nii.gz"
            
            if src_img_path.exists():
                if src_seg_path.exists():
                    dst_img_path = train_images_dir / f"{sub_id}_0000.nii.gz"
                    shutil.copy(src_img_path, dst_img_path)
                    
                    resampled_seg = resample_to_img(
                        source_img=str(src_seg_path),
                        target_img=str(src_img_path),
                        interpolation='nearest'
                    )
                    
                    seg_data = resampled_seg.get_fdata()
                    binary_seg = (seg_data > 0).astype(np.uint8)
                    
                    final_seg = nib.Nifti1Image(binary_seg, resampled_seg.affine, resampled_seg.header)
                    final_seg.set_data_dtype(np.uint8)
                    
                    dst_seg_path = train_labels_dir / f"{sub_id}.nii.gz"
                    nib.save(final_seg, str(dst_seg_path))
                    
                    count += 1
                    if count % 20 == 0:
                        print(f"Synced {count} cases...")

    dataset_json = {
        "channel_names": {"0": "T1"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": count,
        "file_ending": ".nii.gz"
    }
    
    dataset_json_path = nnunet_dataset_dir / "dataset.json"
    with open(dataset_json_path, 'w') as json_file:
        json.dump(dataset_json, json_file, indent=4)

    print("Launching planning and preprocessing...")
    subprocess.run([
        "nnUNetv2_plan_and_preprocess", "-d", str(dataset_id),
        "-pl", "nnUNetPlannerResEncM", "-c", "3d_fullres", "--verify_dataset_integrity"
    ], check=True)

    return None

def main() -> None:
    """
    Main entry point for preparing ATLAS data with command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Prepare ATLAS dataset for nnU-Net.")
    parser.add_argument("--base-dir", type=str, required=True, help="Root path containing 'preprocessed' and 'derivatives' folders.")
    parser.add_argument("--dataset-id", type=int, default=999, help="The nnU-Net dataset identifier.")
    parser.add_argument("--nnunet-raw", type=str, required=True, help="Path for nnUNet_raw.")
    parser.add_argument("--nnunet-preprocessed", type=str, required=True, help="Path for nnUNet_preprocessed.")
    parser.add_argument("--nnunet-results", type=str, required=True, help="Path for nnUNet_results.")
    
    args = parser.parse_args()
    
    os.environ['nnUNet_raw'] = args.nnunet_raw
    os.environ['nnUNet_preprocessed'] = args.nnunet_preprocessed
    os.environ['nnUNet_results'] = args.nnunet_results
    
    prepare_training_data(
        base_dir_path=args.base_dir, 
        dataset_id=args.dataset_id, 
        nnunet_raw_path=args.nnunet_raw
    )
    
    return None

if __name__ == "__main__":
    main()