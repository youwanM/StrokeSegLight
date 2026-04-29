import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import os
import numpy as np
from scipy.stats import wilcoxon
import argparse

def plot_single_axis(ax, df_subset, title, threshold, t_col, s_col, metric_label):
    """Helper function to draw a paired boxplot for a specific metric on a matplotlib axis."""
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

    df_melted = df_subset.melt(id_vars=['Subject'], value_vars=[t_col, s_col], 
                        var_name='Model', value_name=f'{metric_label} Score')

    # Draw Boxplot & Stripplot
    sns.boxplot(x='Model', y=f'{metric_label} Score', hue='Model', data=df_melted, palette="Set2", width=0.4, boxprops={'alpha': 0.6}, legend=False, ax=ax)
    sns.stripplot(x='Model', y=f'{metric_label} Score', data=df_melted, color=".25", alpha=0.2, jitter=True, zorder=0, ax=ax)
    
    student_wins = 0
    teacher_wins = 0
    lines_drawn = 0
    
    # Draw the Paired Lines
    for idx, row in df_subset.iterrows():
        t_val = row[t_col]
        s_val = row[s_col]
        diff = s_val - t_val

        if abs(diff) > threshold:
            lines_drawn += 1
            if diff > 0:
                color = 'green'
                student_wins += 1
            else:
                color = 'red'
                teacher_wins += 1
            
            # Draw connecting line
            ax.plot([0, 1], [t_val, s_val], color=color, alpha=0.2, linewidth=2, zorder=5)

    # Formatting
    sig_text = "Significant" if is_significant else "Not Significant"
    ax.set_title(f'{title}\np-val: {p_value:.4f} ({sig_text})', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(f'{metric_label} Score', fontsize=12)
    ax.set_xlabel('', fontsize=12)
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

def generate_stratified_boxplots_for_metric(df, metric_name, t_col, s_col, threshold, base_filename):
    has_size_cat = 'Mask_Size_Category' in df.columns

    # Setup Figure
    if has_size_cat:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        categories = [
            (f"Total (All Sizes) - {metric_name}", df),
            (f"Small Lesions (S) - {metric_name}", df[df['Mask_Size_Category'] == 'S']),
            (f"Medium Lesions (M) - {metric_name}", df[df['Mask_Size_Category'] == 'M']),
            (f"Large Lesions (L) - {metric_name}", df[df['Mask_Size_Category'] == 'L'])
        ]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 7))
        axes = [axes]
        categories = [(f"Total - {metric_name}", df)]

    stats_report = {}

    # Plot Each Category
    for i, (title, subset) in enumerate(categories):
        stats = plot_single_axis(axes[i], subset, title, threshold, t_col, s_col, metric_name)
        if stats:
            stats_report[title] = stats

    plt.tight_layout(pad=3.0)

    # Add custom legend to the Figure (overall)
    custom_lines = [
        Line2D([0], [0], color='green', lw=2, marker='o', markersize=6, markeredgecolor='black', label=f'Student Won (> {threshold})'),
        Line2D([0], [0], color='red', lw=2, marker='o', markersize=6, markeredgecolor='black', label=f'Teacher Won (> {threshold})')
    ]
    fig.legend(handles=custom_lines, loc='lower center', ncol=2, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    # Save Plot
    out_file = base_filename.replace('.png', f'_{metric_name}.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats_report, out_file

def generate_full_report(csv_path="KD_Experiment_Results.csv", threshold=0.1):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_path}'. Please ensure the script is in the same folder as the CSV.")
        return

    # 1. Load the data
    df = pd.read_csv(csv_path)
    base_outname = csv_path.replace('.csv', '_boxplot.png')

    # Define the metrics we want to analyze
    metrics_to_analyze = [
        {"name": "Dice", "t_col": "Teacher_Dice", "s_col": "Student_Dice"},
        {"name": "Lesion_F1", "t_col": "Teacher_Lesion_F1", "s_col": "Student_Lesion_F1"}
    ]

    all_stats = {}

    for m in metrics_to_analyze:
        # Check if columns exist in the CSV before running
        if m["t_col"] in df.columns and m["s_col"] in df.columns:
            stats, saved_file = generate_stratified_boxplots_for_metric(
                df, m["name"], m["t_col"], m["s_col"], threshold, base_outname
            )
            all_stats[m["name"]] = {"stats": stats, "file": saved_file}
        else:
            print(f"Skipping {m['name']} as columns {m['t_col']}/{m['s_col']} were not found in the CSV.")

    # 2. Print Master Terminal Statistics
    print("\n=============================================")
    print("      STRATIFIED STATISTICS REPORT           ")
    print("=============================================")
    print(f"Threshold for 'Win/Loss' lines: > {threshold}\n")

    for metric_name, data in all_stats.items():
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"            METRIC: {metric_name.upper()}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for title, stats in data["stats"].items():
            print(f"--- {title.upper()} ---")
            print(f"N = {stats['count']} subjects")
            print(f"Mean - Teacher: {stats['teacher_mean']:.4f} | Student: {stats['student_mean']:.4f}")
            print(f"Significant (> {threshold}) Shifts - Student Wins: {stats['student_wins']} | Teacher Wins: {stats['teacher_wins']}")
            
            sig_str = "STATISTICALLY SIGNIFICANT" if stats['is_significant'] else "NO significant difference"
            print(f"Wilcoxon P-Value: {stats['p_value']:.5f} ({sig_str})\n")
        
        print(f"-> Saved plot for {metric_name} to '{data['file']}'\n")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    parser = argparse.ArgumentParser(description="Generate stratified paired boxplots comparing Teacher and Student metrics (Dice & F1).")
    parser.add_argument('--file', type=str, required=True, help="Path to the CSV file containing the metrics and 'Mask_Size_Category'")
    parser.add_argument('--threshold', type=float, default=0.25, help="Difference threshold for drawing colored lines (default: 0.25)")
    args = parser.parse_args()
    
    generate_full_report(
        csv_path=args.file, 
        threshold=args.threshold
    )