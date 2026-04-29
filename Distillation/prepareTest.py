import os
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
root_dir = "ATLAS_R2.1_preprocessed/Training_Preprocessed"
output_dir = "Test_Set_ATLAS_2.1"
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

def split_new_test_data():
    test_count = 0
    # Walk through RXXX folders
    for r_folder in tqdm(os.listdir(root_dir), desc="Scanning R-folders"):
        r_path = os.path.join(root_dir, r_folder)
        if not os.path.isdir(r_path): continue
        
        # Walk through sub-XXXX folders
        for sub in os.listdir(r_path):
            sub_path = os.path.join(r_path, sub, "ses-1", "anat")
            if not os.path.exists(sub_path): continue
            
            # Find metadata
            meta_file = [f for f in os.listdir(sub_path) if f.endswith('metadata.csv')][0]
            df = pd.read_csv(os.path.join(sub_path, meta_file))
            
            # Check for ATLAS 2.0 Testing assignment
            # Note: We check if any cell contains the word 'Testing'
            is_test = df.astype(str).apply(lambda x: x.str.contains('Testing')).any().any()
            
            if is_test:
                # Identify T1 and Mask
                t1 = [f for f in os.listdir(sub_path) if 'T1w.nii.gz' in f][0]
                mask = [f for f in os.listdir(sub_path) if 'mask.nii.gz' in f][0]
                
                # Copy with nnU-Net naming convention
                shutil.copy(os.path.join(sub_path, t1), os.path.join(output_dir, "images", f"{sub}_0000.nii.gz"))
                shutil.copy(os.path.join(sub_path, mask), os.path.join(output_dir, "masks", f"{sub}.nii.gz"))
                test_count += 1

    print(f"\nFinished! Found {test_count} test cases. Files are in '{output_dir}'")

if __name__ == "__main__":
    split_new_test_data()