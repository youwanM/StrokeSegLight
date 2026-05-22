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
    # Get the raw string and remove accidental spaces
    raw_model_name = str(df['Model'].iloc[0]).strip()
    
    # Find the correct key in param_dict by comparing them in lowercase
    matched_key = next((k for k in param_dict.keys() if k.lower() == raw_model_name.lower()), None)
    
    if matched_key is None:
        raise KeyError(f"The model name '{raw_model_name}' from your CSV doesn't match anything in param_dict.")
        
    # Apply the perfectly matched, properly capitalized key
    df['Model'] = matched_key
    df['Params_M'] = param_dict[matched_key]
    
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
marker_palette = {'All': 'o', 'L': 'X', 'M': 's', 'S': '^'} 
sizes_to_plot = ['All', 'L', 'M', 'S']

for i, metric in enumerate(metrics):
    df_sub = df_melted[df_melted['Metric'] == metric]
    df_sub_plot = df_sub[df_sub['Model'] != 'Teacher']
    
    # 1. Plot the main points individually to handle different errorbar rules per category
    for size in sizes_to_plot:
        df_size = df_sub_plot[df_sub_plot['Size'] == size]
        if df_size.empty:
            continue
            
        # No error band for 'All', 95% CI bands for the rest
        errorbar_setting = None if size == 'All' else ('ci', 95)
        
        sns.lineplot(
            data=df_size, 
            x='Params_M', 
            y='Score', 
            color=size_palette[size],
            marker=marker_palette[size],
            linestyle='',                           
            err_style='band',                       
            errorbar=errorbar_setting,
            err_kws={'alpha': 0.2, 'zorder': 1},    
            markersize=10,             
            alpha=0.6,                 
            label=size if i == 0 else None,         
            ax=axes[i],
            zorder=4                                
        )
    
    # 2. Add baseline horizontal lines with ALTERNATING dash logic for overlaps
    teacher_refs = {}
    for size in sizes_to_plot:
        teacher_mask = (df_sub['Model'] == 'Teacher') & (df_sub['Size'] == size)
        if not df_sub[teacher_mask].empty:
            val = df_sub[teacher_mask]['Score'].mean()
            if not pd.isna(val):
                teacher_refs[size] = val
                
    # Group lines that are within 0.01 tolerance of each other
    overlap_groups = []
    for size, val in teacher_refs.items():
        placed = False
        for group in overlap_groups:
            # Check if this value is within 0.01 of the group's first value
            if abs(group[0][1] - val) <= 0.01:
                group.append((size, val))
                placed = True
                break
        if not placed:
            overlap_groups.append([(size, val)])
            
    # Draw the lines, applying the zipper effect for grouped lines
    for group in overlap_groups:
        num_lines = len(group)
        
        # Snap grouped lines to their shared average Y so the zipper interlocks perfectly
        shared_y = sum(v for s, v in group) / num_lines
        
        dash_length = 4
        gap_length = dash_length * (num_lines - 1) if num_lines > 1 else dash_length
        total_cycle = dash_length + gap_length
        
        for idx, (size, original_val) in enumerate(group):
            offset = idx * dash_length
            
            if offset == 0:
                dash_seq = (dash_length, gap_length)
            else:
                dash_seq = (0, offset, dash_length, total_cycle - offset - dash_length)
                
            axes[i].axhline(
                y=shared_y, 
                color=size_palette[size],  
                dashes=dash_seq,
                dash_capstyle='butt', # Keeps ends flat
                linewidth=3,               
                alpha=0.5,                 
                zorder=3 
            )
    
    # 3. Formatting
    axes[i].set_title(titles[i], fontweight='bold', pad=15)
    axes[i].set_xlabel("Number of parameters")
    axes[i].set_ylabel("Score" if i < 3 else "Distance (mm)")
    axes[i].set_xscale('log')
    axes[i].grid(True, which="both", ls="--", alpha=0.5, zorder=0)

# 4. Extract legend from the first subplot and put it at the very top
handles, labels = axes[0].get_legend_handles_labels()

# Placed exactly at the top center
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.98), title="Lesion category")

# Clean up default legend box if generated implicitly
if axes[0].get_legend() is not None:
    axes[0].get_legend().remove()

plt.tight_layout(rect=[0, 0, 1, 0.98])

# 5. Save as PDF
plt.savefig("Metrics_Grid.pdf", format="pdf", bbox_inches='tight', dpi=1200)
print("PDF Plot saved successfully.")