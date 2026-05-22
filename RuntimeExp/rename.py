from pathlib import Path

def add_suffix_to_niftis(root_folder):
    # The string we want to add before the file extension
    suffix_to_add = "_MNI_BET"
    
    # Verify the root folder exists
    root_path = Path(root_folder)
    if not root_path.exists() or not root_path.is_dir():
        print(f"Error: The directory '{root_folder}' does not exist.")
        return

    print(f"Scanning '{root_path}' for NIfTI files...\n")
    print("-" * 60)
    
    count = 0
    # Recursively find all files in the directory
    for file_path in root_path.rglob("*"):
        if not file_path.is_file():
            continue
            
        filename = file_path.name
        
        # Check if it's a NIfTI file and DOES NOT already have the suffix
        if (filename.endswith(".nii.gz") or filename.endswith(".nii")) and suffix_to_add not in filename:
            
            # Determine where to insert the suffix based on the extension
            if filename.endswith(".nii.gz"):
                # Remove the last 7 characters (.nii.gz) and append suffix + extension
                new_filename = filename[:-7] + suffix_to_add + ".nii.gz"
            elif filename.endswith(".nii"):
                # Remove the last 4 characters (.nii) and append suffix + extension
                new_filename = filename[:-4] + suffix_to_add + ".nii"
            
            # Create the full path for the new file
            new_file_path = file_path.with_name(new_filename)
            
            try:
                # Perform the rename
                file_path.rename(new_file_path)
                print(f"RENAMED: {filename}")
                print(f"     TO: {new_filename}\n")
                count += 1
            except Exception as e:
                print(f"ERROR renaming {filename}: {e}")

    print("-" * 60)
    if count == 0:
        print(f"No files needed renaming (they might already have '{suffix_to_add}' or no NIfTI files exist).")
    else:
        print(f"Success! Added '{suffix_to_add}' to {count} file(s).")

if __name__ == "__main__":
    # Prompt the user for the folder to scan
    folder_to_scan = input("Enter the path to the folder containing the files: ")
    
    # Strip quotes in case you drag-and-drop the folder into the terminal
    folder_to_scan = folder_to_scan.strip('"\'') 
    
    # Ask for confirmation before modifying files
    print(f"\nWARNING: This will permanently append '_MNI_BET' to NIfTI files in '{folder_to_scan}'.")
    confirm = input("Are you sure you want to proceed? (y/n): ")
    
    if confirm.lower() == 'y':
        add_suffix_to_niftis(folder_to_scan)
    else:
        print("Operation cancelled.")