"""
Module: prepare_test.py
Description: Extracts testing subjects from the preprocessed ATLAS dataset 
             and formats them according to nnU-Net conventions.
"""

import os
import pandas as pd
import shutil
from tqdm import tqdm

def split_new_test_data(root_dir: str, output_dir: str) -> None:
    """
    Scans the preprocessed ATLAS directory for test subjects and copies their 
    anatomical and mask files to the specified output directory.
    
    Args:
        root_dir (str): The root directory containing preprocessed ATLAS data folders.
        output_dir (str): The destination directory for the formatted test set.
    """
    test_count = 0
    
    images_out_dir = os.path.join(output_dir, "images")
    masks_out_dir = os.path.join(output_dir, "masks")
    
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(masks_out_dir, exist_ok=True)

    for r_folder in tqdm(os.listdir(root_dir), desc="Scanning R-folders"):
        r_path = os.path.join(root_dir, r_folder)
        
        if os.path.isdir(r_path):
            for sub in os.listdir(r_path):
                sub_path = os.path.join(r_path, sub, "ses-1", "anat")
                
                if os.path.exists(sub_path):
                    meta_files = [f for f in os.listdir(sub_path) if f.endswith('metadata.csv')]
                    
                    if meta_files:
                        meta_file = meta_files[0]
                        df = pd.read_csv(os.path.join(sub_path, meta_file))
                        
                        is_test = df.astype(str).apply(lambda x: x.str.contains('Testing')).any().any()
                        
                        if is_test:
                            t1_files = [f for f in os.listdir(sub_path) if 'T1w.nii.gz' in f]
                            mask_files = [f for f in os.listdir(sub_path) if 'mask.nii.gz' in f]
                            
                            if t1_files and mask_files:
                                t1 = t1_files[0]
                                mask = mask_files[0]
                                
                                src_t1_path = os.path.join(sub_path, t1)
                                dst_t1_path = os.path.join(images_out_dir, f"{sub}_0000.nii.gz")
                                shutil.copy(src_t1_path, dst_t1_path)
                                
                                src_mask_path = os.path.join(sub_path, mask)
                                dst_mask_path = os.path.join(masks_out_dir, f"{sub}.nii.gz")
                                shutil.copy(src_mask_path, dst_mask_path)
                                
                                test_count += 1

    print(f"\nFinished! Found {test_count} test cases. Files are saved in '{output_dir}'.")
    
    # SESE Principle: Implicit return at the end of the execution flow
    return None

if __name__ == "__main__":
    # Parameters are localized here to avoid polluting the global scope
    ATLAS_ROOT = "ATLAS_R2.1_preprocessed/Training_Preprocessed"
    TEST_OUTPUT = "Test_Set_ATLAS_2.1"
    
    split_new_test_data(root_dir=ATLAS_ROOT, output_dir=TEST_OUTPUT)