import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import argparse
import os
import itertools

def safe_wilcoxon(df, col1, col2):
    """Safely runs the Wilcoxon test handling edge cases like zero variance."""
    diffs = df[col1] - df[col2]
    
    # Wilcoxon requires at least a few samples and non-zero differences
    if len(df) < 3 or np.all(diffs == 0):
        return 1.0, False, df[col1].mean(), df[col2].mean()
    
    stat, p_value = wilcoxon(df[col1], df[col2])
    is_significant = p_value < 0.05
    return p_value, is_significant, df[col1].mean(), df[col2].mean()

def evaluate_students_pairwise(student_files, student_names, output_csv):
    if len(student_files) != len(student_names):
        print(f"Error: Mismatch in inputs. Provided {len(student_files)} files but {len(student_names)} names.")
        return
    
    if len(student_files) < 2:
        print("Error: You need to provide at least 2 students to perform pairwise comparisons.")
        return

    # 1. Load data
    dfs = {}
    for f, name in zip(student_files, student_names):
        if not os.path.exists(f):
            print(f"Error: File '{f}' could not be found.")
            return
        dfs[name] = pd.read_csv(f)

    # Initialize a list to store results for the CSV
    csv_results = []

    # 2. Generate Pairwise Combinations
    pairs = list(itertools.combinations(student_names, 2))
    
    print("=======================================================")
    print(f"   WILCOXON PAIRED TEST: ALL STUDENTS ({len(pairs)} Pairs)")
    print("=======================================================")

    for name_A, name_B in pairs:
        df_A = dfs[name_A]
        df_B = dfs[name_B]
        
        # Merge on Subject to ensure we are comparing the exact same cases
        merged_df = pd.merge(df_A, df_B, on="Subject", suffixes=('_A', '_B'))
        
        if len(merged_df) == 0:
            print(f"\nError: No matching subjects found between {name_A} and {name_B}.")
            continue

        if 'Mask_Size_Category_A' not in merged_df.columns:
            print(f"Warning: 'Mask_Size_Category' missing. Defaulting to 'Total'.")
            merged_df['Mask_Size_Category_A'] = 'Total'

        # Define strata based on the merged dataframe
        strata = [
            ("OVERALL", merged_df),
            ("SMALL (S)", merged_df[merged_df['Mask_Size_Category_A'] == 'S']),
            ("MEDIUM (M)", merged_df[merged_df['Mask_Size_Category_A'] == 'M']),
            ("LARGE (L)", merged_df[merged_df['Mask_Size_Category_A'] == 'L'])
        ]

        # 3. Define metrics for comparison using the suffixes from the merge
        metrics = [
            ("DICE SCORE", "Student_Dice_A", "Student_Dice_B"),
            ("LESION F1 SCORE", "Student_Lesion_F1_A", "Student_Lesion_F1_B")
        ]

        print(f"\nEvaluating Pair: {name_A.upper()} vs {name_B.upper()}")
        print(f"Total Paired Subjects: {len(merged_df)}\n")

        for metric_name, col_A, col_B in metrics:
            if col_A not in merged_df.columns or col_B not in merged_df.columns:
                print(f"Skipping {metric_name} - columns ({col_A}, {col_B}) not found in data.")
                continue

            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"                 METRIC: {metric_name}")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            for strata_name, subset in strata:
                if len(subset) == 0:
                    print(f"  [{strata_name}] N=0 (No data available)")
                    csv_results.append({
                        "Model_A": name_A,
                        "Model_B": name_B,
                        "Metric": metric_name,
                        "Strata": strata_name,
                        "N": 0,
                        "Mean_A": None,
                        "Mean_B": None,
                        "P_Value": None,
                        "Significant": False
                    })
                    continue

                p_val, is_sig, mean1, mean2 = safe_wilcoxon(subset, col_A, col_B)
                sig_str = "* SIGNIFICANT *" if is_sig else "Not Significant"
                
                print(f"  [{strata_name}] N={len(subset)}")
                print(f"    {name_A} Mean: {mean1:.4f} | {name_B} Mean: {mean2:.4f}")
                print(f"    P-Value: {p_val:.5f} ({sig_str})")
                print("-" * 55)

                # Append data to results list for CSV
                csv_results.append({
                    "Model_A": name_A,
                    "Model_B": name_B,
                    "Metric": metric_name,
                    "Strata": strata_name,
                    "N": len(subset),
                    "Mean_A": mean1,
                    "Mean_B": mean2,
                    "P_Value": p_val,
                    "Significant": is_sig
                })
        
        print("\n" + "="*55)

    # 4. Save results to CSV
    if csv_results:
        results_df = pd.DataFrame(csv_results)
        results_df.to_csv(output_csv, index=False)
        print(f"\n✅ All pairwise results successfully saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Wilcoxon signed-rank tests comparing multiple Student models pairwise and save to CSV.")
    
    parser.add_argument('--students', type=str, nargs='+', required=True, 
                        help="List of paths to student CSV results (e.g., --students file1.csv file2.csv file3.csv)")
    parser.add_argument('--name', type=str, nargs='+', required=True, 
                        help="List of display names for the students (e.g., --name Light Medium Heavy)")
    parser.add_argument('--out_csv', type=str, default="students_pairwise_wilcoxon_results.csv", 
                        help="Path to save the output CSV file (default: students_pairwise_wilcoxon_results.csv)")
    
    args = parser.parse_args()
    
    evaluate_students_pairwise(args.students, args.name, args.out_csv)