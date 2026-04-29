import os
import json
import nibabel as nib
import numpy as np
import subprocess
from pathlib import Path
from nilearn.image import resample_to_img # Requires: pip install nilearn

# 1. Environment Setup
os.environ['nnUNet_raw'] = "/home/ymahe/Desktop/StrokeSegLight/Distillation/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "/home/ymahe/Desktop/StrokeSegLight/Distillation/nnUNet_preprocessed"
os.environ['nnUNet_results'] = "/home/ymahe/Desktop/StrokeSegLight/Distillation/nnUNet_results"

base_dir = Path("/home/ymahe/Desktop/Datasets/ATLAS_2")
images_dir = base_dir / "preprocessed"
labels_dir = base_dir / "derivatives"

dataset_id = 999
nnunet_raw_dir = Path(os.environ['nnUNet_raw']) / f"Dataset{dataset_id}"
train_images = nnunet_raw_dir / "imagesTr"
train_labels = nnunet_raw_dir / "labelsTr"

train_images.mkdir(parents=True, exist_ok=True)
train_labels.mkdir(parents=True, exist_ok=True)

# 2. Process, Resample, and Binarize
subjects = [d for d in images_dir.iterdir() if d.is_dir()]
count = 0

print("Aligning masks to images and binarizing...")
for sub in subjects:
    sub_id = sub.name
    src_img_path = images_dir / sub_id / "anat" / f"{sub_id}_T1w.nii.gz"
    src_seg_path = labels_dir / sub_id / "seg" / f"{sub_id}_seg.nii.gz"
    
    if src_img_path.exists() and src_seg_path.exists():
        # 1. Copy Image
        dst_img = train_images / f"{sub_id}_0000.nii.gz"
        os.system(f"cp {src_img_path} {dst_img}")
        
        # 2. Resample Mask to match Image Header
        # We use nearest neighbor to preserve label values (0 and 1)
        resampled_seg = resample_to_img(
            source_img=str(src_seg_path),
            target_img=str(src_img_path),
            interpolation='nearest'
        )
        
        # 3. Binarize and cast to uint8
        seg_data = resampled_seg.get_fdata()
        binary_seg = (seg_data > 0).astype(np.uint8)
        
        # 4. Save aligned Mask
        final_seg = nib.Nifti1Image(binary_seg, resampled_seg.affine, resampled_seg.header)
        final_seg.set_data_dtype(np.uint8)
        
        dst_seg = train_labels / f"{sub_id}.nii.gz"
        nib.save(final_seg, str(dst_seg))
        
        count += 1
        if count % 20 == 0:
            print(f"Synced {count} cases...")

# 3. Dataset JSON and Preprocessing
dataset_json = {
    "channel_names": {"0": "T1"},
    "labels": {"background": 0, "lesion": 1},
    "numTraining": count,
    "file_ending": ".nii.gz"
}
with open(nnunet_raw_dir / "dataset.json", 'w') as f:
    json.dump(dataset_json, f, indent=4)

print("Launching planning and preprocessing...")
subprocess.run([
    "nnUNetv2_plan_and_preprocess", "-d", str(dataset_id),
    "-pl", "nnUNetPlannerResEncM", "-c", "3d_fullres", "--verify_dataset_integrity"
], check=True)