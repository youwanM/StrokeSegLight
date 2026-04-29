import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import argparse
import os

def safe_wilcoxon(df, col1, col2):
    """Safely runs the Wilcoxon test handling edge cases like zero variance."""
    diffs = df[col1] - df[col2]
    
    # Wilcoxon requires at least a few samples and non-zero differences
    if len(df) < 3 or np.all(diffs == 0):
        return 1.0, False, df[col1].mean(), df[col2].mean()
    
    stat, p_value = wilcoxon(df[col1], df[col2])
    is_significant = p_value < 0.05
    return p_value, is_significant, df[col1].mean(), df[col2].mean()

def evaluate_teacher_vs_students(student_files, student_names, output_csv):
    if len(student_files) != len(student_names):
        print(f"Error: Mismatch in inputs. Provided {len(student_files)} files but {len(student_names)} names.")
        return

    # 1. Load data
    dfs = []
    for f in student_files:
        if not os.path.exists(f):
            print(f"Error: File '{f}' could not be found.")
            return
        dfs.append(pd.read_csv(f))

    if len(dfs) == 0:
        print("Error: No files loaded.")
        return

    # 2. Sanity Check: Ensure teacher metrics match across all CSVs
    print("Running Sanity Check on Teacher metrics across all files...")
    base_df = dfs[0]
    
    teacher_cols = ['Teacher_Dice', 'Teacher_Lesion_F1']
    
    for t_col in teacher_cols:
        if t_col not in base_df.columns:
            print(f"Error: Expected teacher column '{t_col}' not found in {student_files[0]}.")
            return

    for i in range(1, len(dfs)):
        curr_df = dfs[i]
        
        merged = pd.merge(base_df[['Subject'] + teacher_cols], 
                          curr_df[['Subject'] + teacher_cols], 
                          on="Subject", suffixes=('_ref', '_curr'))
        
        if len(merged) != len(base_df) or len(merged) != len(curr_df):
            print(f"Warning: Subject lists differ between '{student_files[0]}' and '{student_files[i]}'.")

        for t_col in teacher_cols:
            is_match = np.allclose(
                merged[f'{t_col}_ref'].fillna(-1), 
                merged[f'{t_col}_curr'].fillna(-1), 
                atol=1e-5
            )
            if not is_match:
                print(f"CRITICAL ERROR: Teacher metric '{t_col}' in '{student_files[i]}' does NOT match the reference '{student_files[0]}'.")
                return
                
    print("-> Sanity Check Passed: Teacher metrics are identical across all CSVs.\n")

    # 3. Define metrics for comparison
    metrics = [
        ("DICE SCORE", "Teacher_Dice", "Student_Dice"),
        ("LESION F1 SCORE", "Teacher_Lesion_F1", "Student_Lesion_F1")
    ]

    # Initialize a list to store results for the CSV
    csv_results = []

    # 4. Generate Report
    print("=======================================================")
    print("      WILCOXON PAIRED TEST: TEACHER VS STUDENTS")
    print("=======================================================")

    for i, (file_path, name) in enumerate(zip(student_files, student_names)):
        df = dfs[i]
        
        if 'Mask_Size_Category' not in df.columns:
            print("Warning: 'Mask_Size_Category' missing. Defaulting to 'Total' for all.")
            df['Mask_Size_Category'] = 'Total'

        strata = [
            ("OVERALL", df),
            ("SMALL (S)", df[df['Mask_Size_Category'] == 'S']),
            ("MEDIUM (M)", df[df['Mask_Size_Category'] == 'M']),
            ("LARGE (L)", df[df['Mask_Size_Category'] == 'L'])
        ]

        print(f"\nEvaluating: Teacher vs {name.upper()}")
        print(f"File: {file_path}")
        print(f"Total Paired Subjects: {len(df)}\n")

        for metric_name, t_col, s_col in metrics:
            if t_col not in df.columns or s_col not in df.columns:
                print(f"Skipping {metric_name} - columns ({t_col}, {s_col}) not found in data.")
                continue

            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"                 METRIC: {metric_name}")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            for strata_name, subset in strata:
                if len(subset) == 0:
                    print(f"  [{strata_name}] N=0 (No data available)")
                    csv_results.append({
                        "Student_Model": name,
                        "Metric": metric_name,
                        "Strata": strata_name,
                        "N": 0,
                        "Teacher_Mean": None,
                        "Student_Mean": None,
                        "P_Value": None,
                        "Significant": False
                    })
                    continue

                p_val, is_sig, mean1, mean2 = safe_wilcoxon(subset, t_col, s_col)
                sig_str = "* SIGNIFICANT *" if is_sig else "Not Significant"
                
                print(f"  [{strata_name}] N={len(subset)}")
                print(f"    Teacher Mean: {mean1:.4f} | {name} Mean: {mean2:.4f}")
                print(f"    P-Value: {p_val:.5f} ({sig_str})")
                print("-" * 55)

                # Append data to results list for CSV
                csv_results.append({
                    "Student_Model": name,
                    "Metric": metric_name,
                    "Strata": strata_name,
                    "N": len(subset),
                    "Teacher_Mean": mean1,
                    "Student_Mean": mean2,
                    "P_Value": p_val,
                    "Significant": is_sig
                })
        
        print("\n" + "="*55)

    # 5. Save results to CSV
    if csv_results:
        results_df = pd.DataFrame(csv_results)
        results_df.to_csv(output_csv, index=False)
        print(f"\n✅ All results successfully saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Wilcoxon signed-rank test comparing a Teacher to multiple Student models and save to CSV.")
    
    parser.add_argument('--students', type=str, nargs='+', required=True, 
                        help="List of paths to student CSV results (e.g., --students file1.csv file2.csv)")
    parser.add_argument('--name', type=str, nargs='+', required=True, 
                        help="List of display names for the students (e.g., --name Light Medium)")
    parser.add_argument('--out_csv', type=str, default="teacher_vs_students_wilcoxon_results.csv", 
                        help="Path to save the output CSV file (default: teacher_vs_students_wilcoxon_results.csv)")
    
    args = parser.parse_args()
    
    evaluate_teacher_vs_students(args.students, args.name, args.out_csv)