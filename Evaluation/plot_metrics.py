"""
Module: plot_metrics.py
Description: Generates statistical visualizations for model evaluation, 
             including paired boxplots (Teacher vs Student) and capacity 
             scaling plots (metrics across parameter counts).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

# Global visual configuration for Matplotlib
plt.rcParams.update({
    'font.size': 16, 
    'axes.titlesize': 20, 
    'axes.labelsize': 18, 
    'xtick.labelsize': 16, 
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18
})

def generate_paired_boxplots(teacher_csv: str, student_csv: str, output_path: str, threshold: float = 0.1) -> None:
    """
    Generates a 4x4 grid of stratified paired boxplots comparing Teacher and Student metrics.
    Calculates Wilcoxon signed-rank tests for statistical significance.
    """
    if os.path.exists(teacher_csv) and os.path.exists(student_csv):
        df_teacher = pd.read_csv(teacher_csv)
        df_student = pd.read_csv(student_csv)
        
        student_name = df_student['Model'].iloc[0] if 'Model' in df_student.columns else "Student"
        
        merge_cols = ["Subject"]
        if "Size" in df_teacher.columns and "Size" in df_student.columns:
            merge_cols.append("Size")
            
        df_merged = pd.merge(df_teacher, df_student, on=merge_cols, suffixes=('_Teacher', '_Student'))
        
        if "Size" not in df_merged.columns:
            if "Size_Teacher" in df_merged.columns:
                df_merged["Size"] = df_merged["Size_Teacher"]
            elif "Size_Student" in df_merged.columns:
                df_merged["Size"] = df_merged["Size_Student"]
            else:
                df_merged["Size"] = "Unknown"

        metrics_to_analyze = [
            {"name": "Global Dice", "col_base": "Global_Dice"},
            {"name": "Lesion Dice", "col_base": "Lesion_Dice"},
            {"name": "Lesion F1",   "col_base": "Lesion_F1"},
            {"name": "Average Surface Distance (mm)", "col_base": "Average_Surface_Distance_mm"}
        ]

        size_categories = [
            {"label": "Total (All Sizes)", "val": None},
            {"label": "Large Lesions (L)", "val": "L"},
            {"label": "Medium Lesions (M)", "val": "M"},
            {"label": "Small Lesions (S)", "val": "S"}
        ]

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        
        for row_idx, cat in enumerate(size_categories):
            cat_label = cat["label"]
            cat_val = cat["val"]
            
            df_subset = df_merged if cat_val is None else df_merged[df_merged['Size'] == cat_val]

            for col_idx, m in enumerate(metrics_to_analyze):
                ax = axes[row_idx, col_idx]
                t_col = f'{m["col_base"]}_Teacher'
                s_col = f'{m["col_base"]}_Student'
                
                if t_col in df_merged.columns and s_col in df_merged.columns:
                    df_valid = df_subset.dropna(subset=[t_col, s_col]).copy()
                    
                    if not df_valid.empty:
                        diffs = df_valid[s_col] - df_valid[t_col]
                        
                        if len(df_valid) < 3 or np.all(diffs == 0):
                            p_value = 1.0
                            is_sig = False
                        else:
                            _, p_value = wilcoxon(df_valid[t_col], df_valid[s_col])
                            is_sig = p_value < 0.05
                            
                        df_melted = df_valid.melt(
                            id_vars=['Subject'], 
                            value_vars=[t_col, s_col], 
                            var_name='Model', 
                            value_name=f'{m["name"]} Score'
                        )
                        df_melted['Model'] = df_melted['Model'].replace({t_col: 'Teacher', s_col: student_name})
                        
                        sns.boxplot(x='Model', y=f'{m["name"]} Score', hue='Model', data=df_melted, 
                                    palette={"Teacher": "#8da0cb", student_name: "#fc8d62"}, 
                                    width=0.4, boxprops={'alpha': 0.6}, legend=False, ax=ax)
                        sns.stripplot(x='Model', y=f'{m["name"]} Score', data=df_melted, color=".25", alpha=0.3, jitter=True, zorder=0, ax=ax)
                        
                        sig_text = "Significant" if is_sig else "Not Significant"
                        ax.set_title(f"{m['name']} - {cat_label}\np-val: {p_value:.4f} ({sig_text})", fontsize=11)
                        
                        if "Average Surface Distance" in m["name"]:
                            ax.set_ylim(auto=True)
                            current_vals = df_valid[[t_col, s_col]].values.flatten()
                            current_vals = current_vals[~np.isnan(current_vals)]
                            if len(current_vals) > 0:
                                ax.set_ylim(0.05, np.percentile(current_vals, 75) * 1.5)
                        else:
                            ax.set_ylim(-0.05, 1.05)
                            
                        ax.set_ylabel(f'{m["name"]} Score', fontsize=10)
                        ax.set_xlabel('', fontsize=10)
                        ax.grid(axis='y', linestyle='--', alpha=0.5)
                    else:
                        ax.set_title(f"{m['name']} - {cat_label}\n(No Data)", fontsize=12)
                        ax.axis('off')
                else:
                    ax.set_title(f"{m['name']}\n(Data Missing)", fontsize=12)
                    ax.axis('off')

        plt.tight_layout(pad=4.0)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"-> Saved 4x4 paired boxplot grid to '{output_path}'")
        
    else:
        print("Error: Could not find one or both of the provided CSV files.")
        
    return None

def main() -> None:
    """
    Main entry point for generating evaluation plots using command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate paired boxplots comparing Teacher and Student metrics.")
    parser.add_argument("--teacher-csv", type=str, required=True, help="Path to the Teacher evaluation CSV file.")
    parser.add_argument("--student-csv", type=str, required=True, help="Path to the Student evaluation CSV file.")
    parser.add_argument("--output-path", type=str, default="evaluation_results/Teacher_vs_Student_Boxplots.png", help="Path to save the output plot image.")
    
    args = parser.parse_args()
    
    generate_paired_boxplots(
        teacher_csv=args.teacher_csv,
        student_csv=args.student_csv,
        output_path=args.output_path
    )
    return None

if __name__ == "__main__":
    main()