import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make all text, titles, and ticks globally larger
plt.rcParams.update({
    'font.size': 16, 
    'axes.titlesize': 20, 
    'axes.labelsize': 18, 
    'xtick.labelsize': 16, 
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18
})

# Param dictionary (multiplied by 1,000,000 so the axis represents absolute numbers)
param_dict = {
    'Femto': 0.05 * 1e6,
    'Pico': 0.21 * 1e6,
    'Nano': 0.84 * 1e6,
    'ExtraExtraLight': 2.61 * 1e6,
    'ExtraLight': 4.72 * 1e6,
    'Light': 10.44 * 1e6,
    'Small': 16.39 * 1e6,
    'Medium': 35.27 * 1e6,
    'Large':  52.91 * 1e6,
    'Teacher': 102.35 * 1e6
}

# Load the data
DATA = [
    pd.read_csv("Subject_Detection_Femto_20260504_080325.csv"),
    pd.read_csv("Subject_Detection_Pico_20260504_080702.csv"),
    pd.read_csv("Subject_Detection_Nano_20260504_081039.csv"),
    pd.read_csv("Subject_Detection_ExtraExtraLight_20260504_081412.csv"),
    pd.read_csv("Subject_Detection_ExtraLight_20260504_081752.csv"),
    pd.read_csv("Subject_Detection_Light_20260504_082141.csv"),
    pd.read_csv("Subject_Detection_Small_20260504_082557.csv"),
    pd.read_csv("Subject_Detection_Medium_20260504_083028.csv"),
    pd.read_csv("Subject_Detection_Large_20260504_083515.csv"),
    pd.read_csv("Subject_Detection_Teacher_20260504_075941.csv")
]

dfs = []
for df in DATA:
    model_name = df['Model'].iloc[0]
    df['Params_M'] = param_dict[model_name]
    df_all = df.copy()
    df_all['Size'] = 'All'
    df_comb = pd.concat([df, df_all], ignore_index=True)
    dfs.append(df_comb)

df_total = pd.concat(dfs, ignore_index=True)
metrics = ['Global_Dice', 'Lesion_Dice', 'Lesion_F1', 'Average_Surface_Distance_mm']
df_melted = df_total.melt(id_vars=['Subject', 'Model', 'Params_M', 'Size'], value_vars=metrics, var_name='Metric', value_name='Score')

# Larger vertical format
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
titles = ['Global Dice Score', 'Lesion Dice Score', 'Lesion F1 Score', 'Average Surface Distance (mm)']

size_palette = {'All': 'black', 'L': '#1f77b4', 'M': '#ff7f0e', 'S': '#2ca02c'}
sizes_to_plot = ['All', 'L', 'M', 'S']

for i, metric in enumerate(metrics):
    df_sub = df_melted[df_melted['Metric'] == metric]
    df_sub_plot = df_sub[df_sub['Model'] != 'Teacher']
    
    # 1. Plot the main points
    sns.lineplot(
        data=df_sub_plot, 
        x='Params_M', 
        y='Score', 
        hue='Size', 
        hue_order=sizes_to_plot,
        palette=size_palette,
        style='Size',
        style_order=sizes_to_plot,
        markers=True,
        dashes=False,
        linestyle='',
        err_style='bars',
        errorbar=('ci', 95),
        err_kws={'linewidth': 3},  # Make the error bar lines thicker
        markersize=14,             # Enlarge the scatter points
        alpha=0.6,                 # Add opacity to see overlapping elements
        legend=(i == 0),           # Only generate legend for the first plot to grab handles
        ax=axes[i]
    )
    
    # 2. Add horizontal dashed reference lines for Teacher
    for size in sizes_to_plot:
        teacher_mask = (df_sub['Model'] == 'Teacher') & (df_sub['Size'] == size)
        if not df_sub[teacher_mask].empty:
            teacher_ref = df_sub[teacher_mask]['Score'].mean()
            axes[i].axhline(
                y=teacher_ref, 
                color=size_palette[size],  
                linestyle='--', 
                linewidth=3,               
                alpha=0.7,                 
                zorder=0                   
            )
    
    # 3. Formatting
    axes[i].set_title(titles[i], fontweight='bold', pad=15)
    axes[i].set_xlabel("Number of Parameters")
    axes[i].set_ylabel("Score" if i < 3 else "Distance (mm)")
    axes[i].set_xscale('log')
    axes[i].grid(True, which="both", ls="--", alpha=0.5)

# 4. Extract legend from the first subplot and put it at the very top
handles, labels = axes[0].get_legend_handles_labels()

# INCREASED the y-value in bbox_to_anchor from 1.01 to 1.04 to push it up higher
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0), title="Lesion category")

# Remove the default legend from the first subplot
axes[0].get_legend().remove()

plt.tight_layout(rect=[0, 0, 1, 0.98])

# 5. Save as PDF
plt.savefig("Metrics_Grid.pdf", format="pdf", bbox_inches='tight', dpi=1200)
print("PDF Plot saved successfully.")