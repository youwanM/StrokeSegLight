"""
Module: run_benchmark.py
Description: Orchestrates batch inference runs of the StrokeSeg2 application 
             on NIfTI files across different model variants and precisions.
"""

import os
import time
import shutil
import argparse
import subprocess
from pathlib import Path

def clear_directory(target_path: Path) -> None:
    """
    Safely clears all contents of a given directory.
    """
    if target_path.exists():
        for item in target_path.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {item}. ({e})")
    else:
        target_path.mkdir(parents=True, exist_ok=True)
    return None

def clean_preproc_files(root_path: Path) -> int:
    """
    Removes intermediate PREPROC.nii.gz files recursively from the input directory.
    """
    removed_count = 0
    for f in root_path.rglob("*"):
        try:
            if f.is_file() and f.name.endswith('PREPROC.nii.gz'):
                f.unlink()
                removed_count += 1
        except Exception as e:
            print(f"Warning: Could not delete {f}. ({e})")
    return removed_count

def process_nifti_files(root_folder: str, model_name: str, exe_path: str, output_dir: str, app_log_dir: str, final_log_destination: str) -> None:
    """
    Runs batch inference over all NIfTI files in the root folder using the specified model.
    """
    output_path = Path(output_dir)
    log_path = Path(app_log_dir)
    root_path = Path(root_folder)
    
    print(f"Clearing output folder: {output_path}...")
    clear_directory(output_path)
    
    print(f"Clearing old logs in: {log_path}...")
    clear_directory(log_path)
    
    if root_path.exists() and root_path.is_dir():
        print(f"Removing preprocessed files in '{root_path}'...")
        clean_preproc_files(root_path)

        nifti_files = [f for f in root_path.rglob("*") if f.name.endswith(('.nii', '.nii.gz'))]

        if nifti_files:
            print(f"Found {len(nifti_files)} file(s). Starting processing for model: {model_name}...\n")
            print("-" * 50)
            
            time.sleep(5) 
            
            executable_found = True
            
            for nifti_file in nifti_files:
                if executable_found:
                    file_path_str = str(nifti_file.resolve())
                    print(f"Processing: {file_path_str}")
                    
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
                    except subprocess.CalledProcessError as e:
                        print(f"FAILED: An error occurred while processing {nifti_file.name}. ({e})\n")
                    except FileNotFoundError:
                        print(f"CRITICAL ERROR: Could not find the executable at {exe_path}")
                        executable_found = False
                        
                    print("-" * 50)
                
            if executable_found:
                try:
                    shutil.copytree(log_path, final_log_destination, dirs_exist_ok=True)
                    print(f"SUCCESS: Logs backed up to {final_log_destination}")
                except Exception as e:
                    print(f"FAILED to backup logs: {e}")
                    
            clean_preproc_files(root_path)
        else:
            print("No NIfTI files found in the specified directory.")
    else:
        print(f"Error: The directory '{root_folder}' does not exist.")
        
    return None

def main() -> None:
    """
    Main entry point for running the benchmark script with command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run batch inference benchmark for StrokeSeg2.")
    parser.add_argument("--exe-path", type=str, required=True, help="Absolute path to the StrokeSeg2 executable.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the input NIfTI files.")
    parser.add_argument("--output-dir", type=str, default="out", help="Directory where inference outputs will be saved.")
    parser.add_argument("--log-dest-dir", type=str, default="logs", help="Destination base directory for saving benchmark logs.")
    parser.add_argument("--models", type=str, nargs='+', default=["Teacher_fp32", "Nano_fp32", "Teacher_fp16", "Nano_fp16"], help="List of model variants to benchmark.")
    
    args = parser.parse_args()
    
    appdata_path = os.environ.get("APPDATA", "")
    if appdata_path:
        app_log_dir = os.path.join(appdata_path, "Empenn - INRIA", "StrokeSeg2")
    else:
        app_log_dir = "app_logs" 
        
    for model in args.models:
        final_dest = os.path.join(args.log_dest_dir, f"{model}_logs")
        process_nifti_files(
            root_folder=args.input_dir,
            model_name=model,
            exe_path=args.exe_path,
            output_dir=args.output_dir,
            app_log_dir=app_log_dir,
            final_log_destination=final_dest
        )
        
    return None

if __name__ == "__main__":
    main()