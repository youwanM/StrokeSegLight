import subprocess
from pathlib import Path
import time
import shutil

def process_nifti_files(root_folder, model_name):
    # 1. Define your executables and fixed arguments
    exe_path = r"C:\Users\z0051vdu\source\repos\strokeseg2-app-build\Release\strokeseg2-app.exe"
    output_dir = r"C:\d\out"
    app_log_dir = Path(r"C:\Users\z0051vdu\AppData\Roaming\Empenn - INRIA\StrokeSeg2")
    final_log_destination = f"C:\d\{model_name}_logs"
    
    # Clear output folder before processing
    output_path = Path(output_dir)
    print(f"Clearing output folder: {output_path}...")
    if output_path.exists():
        for item in output_path.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {item}. It might be in use. ({e})")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    print("Output folder cleared.\n")

    print(f"Clearing old logs in: {app_log_dir}...")
    if app_log_dir.exists():
        for item in app_log_dir.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {item.name}. It might be in use. ({e})")
    else:
        app_log_dir.mkdir(parents=True, exist_ok=True)
    print("Log folder cleared.\n")
    
    # 2. Verify the root folder exists
    root_path = Path(root_folder)
    if not root_path.exists() or not root_path.is_dir():
        print(f"Error: The directory '{root_folder}' does not exist.")
        return

    # 3. Recursively find all .nii and .nii.gz files
    # Remove all files ending with PREPROC.nii.gz recursively in the input folder
    print(f"Removing files ending with 'PREPROC.nii.gz' in '{root_path}'...")
    removed_count = 0
    for f in root_path.rglob("*"):
        try:
            if f.is_file() and f.name.endswith('PREPROC.nii.gz'):
                f.unlink()
                removed_count += 1
        except Exception as e:
            print(f"Warning: Could not delete {f}. ({e})")
    print(f"Removed {removed_count} PREPROC.nii.gz file(s).")

    print(f"Scanning '{root_path}' for NIfTI files...")
    nifti_files = [f for f in root_path.rglob("*") if f.name.endswith(('.nii', '.nii.gz'))]

    if not nifti_files:
        print("No NIfTI files found in the specified directory.")
        return

    print(f"Found {len(nifti_files)} file(s). Starting processing...\n")
    print("-" * 50)
    
    # Wait for JoularCore to initialize before processing files
    time.sleep(5) 
    # 4. Iterate through each file and run the command
    for nifti_file in nifti_files:
        file_path_str = str(nifti_file.resolve())
        print(f"Processing: {file_path_str}")
        
        # Build the command array
        command = [
            exe_path,
            "--input", file_path_str,
            "-o", output_dir,
            "--model", model_name,
            "--verbose",
            "--skip-preproc"
        ]
        try:
            subprocess.run(command, check=True)
            print(f"SUCCESS: Finished processing {nifti_file.name}\n")
            print("-" * 50)
            
        except subprocess.CalledProcessError as e:
            print(f"FAILED: An error occurred while processing {nifti_file.name}.")
            print(f"Error details: {e}\n")
            print("-" * 50)
        except FileNotFoundError:
            print(f"CRITICAL ERROR: Could not find the executable at {exe_path}")
            print("Please check the path and try again.")
            break # Stop the loop if the exe itself is missing
    try:
            # dirs_exist_ok=True ensures it won't crash if you run the same model name twice
            shutil.copytree(app_log_dir, final_log_destination, dirs_exist_ok=True)
            print(f"SUCCESS: Logs backed up to {final_log_destination}")
    except Exception as e:
            print(f"FAILED to backup logs: {e}")
    # Remove any files ending with PREPROC.nii.gz in the input folder after processing
    try:
        removed_end_count = 0
        for f in root_path.rglob("*"):
            try:
                if f.is_file() and f.name.endswith('PREPROC.nii.gz'):
                    f.unlink()
                    removed_end_count += 1
            except Exception as e:
                print(f"Warning: Could not delete {f}. ({e})")
        print(f"Removed {removed_end_count} PREPROC.nii.gz file(s) from input folder after processing.")
    except Exception as e:
        print(f"Failed to remove PREPROC.nii.gz files at end: {e}")
    
if __name__ == "__main__":
    # Use a raw string (r"...") for Windows paths to prevent escape character errors
    folder_to_scan = r"C:\d\Test_Set_ATLAS_2.1\images"
    
    process_nifti_files(folder_to_scan, "Teacher_fp32")
    process_nifti_files(folder_to_scan, "Nano_fp32")
    process_nifti_files(folder_to_scan, "Teacher_fp16")
    process_nifti_files(folder_to_scan, "Nano_fp16")