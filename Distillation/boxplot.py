import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import os
import numpy as np
from scipy.stats import wilcoxon
import argparse

def plot_single_axis(ax, df_subset, title, threshold, t_col, s_col, metric_label, student_name):
    """Helper function to draw a paired boxplot for a specific metric on a matplotlib axis."""
    df_subset = df_subset.dropna(subset=[t_col, s_col]).copy()

    if len(df_subset) == 0:
        ax.set_title(f"{title}\n(No Data)", fontsize=12)
        ax.axis('off')
        return None

    # Handle Wilcoxon if all differences are exactly zero or sample size too small
    diffs = df_subset[s_col] - df_subset[t_col]

    if len(df_subset) < 3 or np.all(diffs == 0):
        p_value = 1.0
        is_significant = False
    else:
        stat, p_value = wilcoxon(df_subset[t_col], df_subset[s_col])
        is_significant = p_value < 0.05

    # Melt dataframe for seaborn boxplot compatibility
    df_melted = df_subset.melt(
        id_vars=['Subject'], 
        value_vars=[t_col, s_col], 
        var_name='Model', 
        value_name=f'{metric_label} Score'
    )
    
    # Rename variables for cleaner axis labels
    df_melted['Model'] = df_melted['Model'].replace({t_col: 'Teacher', s_col: student_name})

    # Draw Boxplot & Stripplot
    sns.boxplot(x='Model', y=f'{metric_label} Score', hue='Model', data=df_melted, 
                palette={"Teacher": "#8da0cb", student_name: "#fc8d62"}, 
                width=0.4, boxprops={'alpha': 0.6}, legend=False, ax=ax)
    sns.stripplot(x='Model', y=f'{metric_label} Score', data=df_melted, color=".25", alpha=0.3, jitter=True, zorder=0, ax=ax)
    
    student_wins = 0
    teacher_wins = 0
    lines_drawn = 0
    
    # Draw the Paired Lines
    '''
    for idx, row in df_subset.iterrows():
        t_val = row[t_col]
        s_val = row[s_col]
        diff = s_val - t_val

        if abs(diff) > threshold:
            lines_drawn += 1
            if diff > 0:
                color = 'green'  # Student improved
                student_wins += 1
            else:
                color = 'red'    # Teacher was better
                teacher_wins += 1
            
            # Draw connecting line
            ax.plot([0, 1], [t_val, s_val], color=color, alpha=0.25, linewidth=2, zorder=5)
    '''

    # Formatting
    sig_text = "Significant" if is_significant else "Not Significant"
    ax.set_title(f'{title}\np-val: {p_value:.4f} ({sig_text})', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(f'{metric_label} Score', fontsize=10)
    ax.set_xlabel('', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Return stats for terminal report
    return {
        'count': len(df_subset),
        'teacher_mean': df_subset[t_col].mean(),
        'student_mean': df_subset[s_col].mean(),
        'student_wins': student_wins,
        'teacher_wins': teacher_wins,
        'p_value': p_value,
        'is_significant': is_significant
    }

def generate_full_report(teacher_csv, student_csv, threshold=0.1):
    if not os.path.exists(teacher_csv) or not os.path.exists(student_csv):
        print("Error: Could not find one or both of the provided CSV files.")
        return

    # 1. Load data
    df_teacher = pd.read_csv(teacher_csv)
    df_student = pd.read_csv(student_csv)
    
    # Identify student name
    student_name = df_student['Model'].iloc[0] if 'Model' in df_student.columns else "Student"

    # Smart Merge: If 'Size' exists in both, merge on it so we don't get Size_x and Size_y
    merge_cols = ["Subject"]
    if "Size" in df_teacher.columns and "Size" in df_student.columns:
        merge_cols.append("Size")
        
    df_merged = pd.merge(df_teacher, df_student, on=merge_cols, suffixes=('_Teacher', '_Student'))

    # Fallback: If Size was only in one file, align it explicitly to a master 'Size' column
    if "Size" not in df_merged.columns:
        if "Size_Teacher" in df_merged.columns:
            df_merged["Size"] = df_merged["Size_Teacher"]
        elif "Size_Student" in df_merged.columns:
            df_merged["Size"] = df_merged["Size_Student"]
        else:
            print("Warning: No 'Size' column found in the CSV files. Cannot stratify by size.")
            df_merged["Size"] = "Unknown"

    # 2. Setup Metrics and Categories
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

    # Setup Figure (4 Rows x 4 Columns)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    all_stats = {}

    # 3. Generate Plot Grid
    for row_idx, cat in enumerate(size_categories):
        cat_label = cat["label"]
        cat_val = cat["val"]
        
        # Filter the dataframe for this row
        if cat_val is None:
            df_subset = df_merged
        else:
            df_subset = df_merged[df_merged['Size'] == cat_val]

        all_stats[cat_label] = {}

        for col_idx, m in enumerate(metrics_to_analyze):
            ax = axes[row_idx, col_idx]
            t_col = f'{m["col_base"]}_Teacher'
            s_col = f'{m["col_base"]}_Student'
            
            # Check columns
            if t_col in df_merged.columns and s_col in df_merged.columns:
                title = f"{m['name']} - {cat_label}"
                stats = plot_single_axis(
                    ax=ax, 
                    df_subset=df_subset, 
                    title=title, 
                    threshold=threshold, 
                    t_col=t_col, 
                    s_col=s_col, 
                    metric_label=m['name'],
                    student_name=student_name
                )
                all_stats[cat_label][m["name"]] = stats
            else:
                ax.set_title(f"{m['name']}\n(Data Missing)", fontsize=12)
                ax.axis('off')
            # Adjust Y-axis for ASD (not bounded by 0-1)
            if "Average Surface Distance" in m["name"]:
                ax.set_ylim(auto=True)
                # Add a small margin to the top of the dynamic range
                current_vals = df_subset[[t_col, s_col]].values.flatten()
                current_vals = current_vals[~np.isnan(current_vals)]
                if len(current_vals) > 0:
                    ax.set_ylim(0.05, np.percentile(current_vals, 75) * 1.5)

    plt.tight_layout(pad=4.0)

    # 4. Add custom legend to the Figure
    #custom_lines = [
    #    Line2D([0], [0], color='green', lw=2, marker='o', markersize=6, markeredgecolor='black', label=f'{student_name} Won (> {threshold})'),
    #    Line2D([0], [0], color='red', lw=2, marker='o', markersize=6, markeredgecolor='black', label=f'Teacher Won (> {threshold})')
    #]
    #fig.legend(handles=custom_lines, loc='lower center', ncol=2, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    # Save Plot
    base_outname = student_csv.replace('.csv', '_Stratified_Comparison.png')
    plt.savefig(base_outname, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Print Master Terminal Statistics
    print("\n=======================================================")
    print(f"  STRATIFIED EVALUATION: TEACHER vs {student_name.upper()}")
    print("=======================================================")
    print(f"Threshold for 'Win/Loss' lines: > {threshold}\n")

    for cat_label, metrics_dict in all_stats.items():
        print(f"{cat_label.upper()} ")
        
        for metric_name, stats in metrics_dict.items():
            if stats is None: continue
            print(f"--- {metric_name} ---")
            print(f"  N = {stats['count']} subjects")
            print(f"  Mean - Teacher: {stats['teacher_mean']:.4f} | {student_name}: {stats['student_mean']:.4f}")
            print(f"  Significant (> {threshold}) Shifts - {student_name} Wins: {stats['student_wins']} | Teacher Wins: {stats['teacher_wins']}")
            
            sig_str = "SIGNIFICANT" if stats['is_significant'] else "NOT significant"
            print(f"  Wilcoxon P-Value: {stats['p_value']:.5f} ({sig_str})\n")
    
    print(f"-> Saved 4x4 plot grid to '{base_outname}'\n")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    parser = argparse.ArgumentParser(description="Generate 4x4 stratified paired boxplots comparing Teacher and Student metrics.")
    parser.add_argument('--teacher', type=str, required=True, help="Path to the Teacher CSV file")
    parser.add_argument('--student', type=str, required=True, help="Path to the Student CSV file")
    parser.add_argument('--threshold', type=float, default=0.1, help="Difference threshold for drawing colored lines (default: 0.1)")
    args = parser.parse_args()
    
    generate_full_report(
        teacher_csv=args.teacher, 
        student_csv=args.student,
        threshold=args.threshold
    )