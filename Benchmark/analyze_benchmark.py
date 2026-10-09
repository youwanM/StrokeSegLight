"""
Module: analyze_benchmark.py
Description: Parses application logs and hardware system power CSV files 
             to compute inference time, energy consumption, and performance ratios.
"""

import os
import glob
import re
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

def parse_log_file(filepath: str) -> tuple:
    """
    Parses log files to extract inference start time, end time, and total duration.
    """
    start_time = None
    end_time = None
    duration_sec = None
    pattern = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")
    
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        for line in lines:
            if start_time is None or end_time is None:
                if "Running inference on patches" in line and start_time is None:
                    match = pattern.match(line)
                    if match:
                        try:
                            start_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
                        except ValueError:
                            pass
                elif "Normalizing" in line and end_time is None:
                    match = pattern.match(line)
                    if match:
                        try:
                            end_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
                        except ValueError:
                            pass
                            
        if start_time is not None and end_time is not None:
            duration_sec = (end_time - start_time).total_seconds()
            
    return start_time, end_time, duration_sec

def compute_energy_summary(root_dir: str, baseline_power: float, output_csv: str) -> None:
    """
    Aggregates hardware logs and power CSVs to compute runtime, power, and energy summaries.
    """
    results = []
    print("Scanning directories for Hardware CSVs and logs...\n")
    
    if os.path.exists(root_dir):
        for hw_dir in os.listdir(root_dir):
            hw_path = os.path.join(root_dir, hw_dir)
            if os.path.isdir(hw_path):
                csv_files = glob.glob(os.path.join(hw_path, "*.csv"))
                if csv_files:
                    hw_csv_path = csv_files[0]
                    df_valid = False
                    
                    try:
                        df = pd.read_csv(hw_csv_path, encoding='latin1', on_bad_lines='skip', low_memory=False)
                        power_col = next((col for col in df.columns if "Total System Power" in col), None)
                        if power_col:
                            datetime_str = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
                            df['Datetime'] = pd.to_datetime(datetime_str, format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')
                            df[power_col] = pd.to_numeric(df[power_col], errors='coerce')
                            df = df.dropna(subset=['Datetime', power_col]).sort_values('Datetime')
                            df_valid = True
                    except Exception as e:
                        print(f"  -> Error loading {hw_csv_path}: {e}")
                        
                    if df_valid:
                        for model_dir in os.listdir(hw_path):
                            model_dir_path = os.path.join(hw_path, model_dir)
                            if os.path.isdir(model_dir_path):
                                log_files = glob.glob(os.path.join(model_dir_path, "*.log"))
                                if log_files:
                                    duration_samples = []
                                    total_power_samples = []
                                    valid_inferences = 0
                                    
                                    for log_file in log_files:
                                        start_time, end_time, duration = parse_log_file(log_file)
                                        
                                        if start_time and end_time:
                                            mask = (df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)
                                            power_values = df.loc[mask, power_col]
                                            
                                            if power_values.empty:
                                                closest_idx = (df['Datetime'] - start_time).abs().idxmin()
                                                total_power_samples.append(df.loc[closest_idx, power_col])
                                            else:
                                                total_power_samples.extend(power_values.tolist())
                                                
                                            duration_samples.append(duration)
                                            valid_inferences += 1
                                            
                                    if valid_inferences > 0:
                                        avg_duration = np.mean(duration_samples)
                                        std_duration = np.std(duration_samples, ddof=1) if len(duration_samples) > 1 else 0.0
                                        avg_power = np.mean(total_power_samples) if total_power_samples else 0.0
                                        std_power = np.std(total_power_samples, ddof=1) if len(total_power_samples) > 1 else 0.0
                                        
                                        results.append({
                                            'Hardware': hw_dir,
                                            'Model_Precision': model_dir,
                                            'Inferences_Run': valid_inferences,
                                            'Avg_Inference_Time (sec)': avg_duration,
                                            'Time_StdDev (sec)': std_duration,
                                            'Avg_System_Power (W)': avg_power,
                                            'Power_StdDev (W)': std_power
                                        })

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by=['Hardware', 'Model_Precision'], ascending=[True, False]).reset_index(drop=True)

        results_df['Corrected_Avg_System_Power (W)'] = results_df['Avg_System_Power (W)'] - baseline_power
        results_df['Energy (J)'] = results_df['Corrected_Avg_System_Power (W)'] * results_df['Avg_Inference_Time (sec)']
        results_df['Energy_Uncertainty (J)'] = np.sqrt(
            (results_df['Avg_Inference_Time (sec)'] * results_df['Power_StdDev (W)'])**2 + 
            (results_df['Corrected_Avg_System_Power (W)'] * results_df['Time_StdDev (sec)'])**2
        )

        max_power = results_df['Corrected_Avg_System_Power (W)'].max()
        max_energy = results_df['Energy (J)'].max()
        
        results_df['Ratio Vs Max (Power)'] = (max_power - results_df['Corrected_Avg_System_Power (W)']) / max_power
        results_df['Ratio Vs Max (Power) Uncertainty'] = results_df['Power_StdDev (W)'] / max_power
        results_df['Ratio Vs Max (Energy)'] = (max_energy - results_df['Energy (J)']) / max_energy
        results_df['Ratio Vs Max (Energy) Uncertainty'] = results_df['Energy_Uncertainty (J)'] / max_energy
        
        cols = [
            'Hardware', 'Model_Precision', 'Inferences_Run', 
            'Avg_Inference_Time (sec)', 'Time_StdDev (sec)', 
            'Avg_System_Power (W)', 'Power_StdDev (W)', 'Corrected_Avg_System_Power (W)', 
            'Ratio Vs Max (Power)', 'Ratio Vs Max (Power) Uncertainty',
            'Energy (J)', 'Energy_Uncertainty (J)', 
            'Ratio Vs Max (Energy)', 'Ratio Vs Max (Energy) Uncertainty'
        ]
        results_df = results_df[cols]

        results_df['Avg_Inference_Time (sec)'] = results_df['Avg_Inference_Time (sec)'].map('{:.2f}'.format)
        results_df['Time_StdDev (sec)'] = results_df['Time_StdDev (sec)'].map('{:.2f}'.format)
        results_df['Avg_System_Power (W)'] = results_df['Avg_System_Power (W)'].map('{:.1f}'.format)
        results_df['Power_StdDev (W)'] = results_df['Power_StdDev (W)'].map('{:.1f}'.format)
        results_df['Corrected_Avg_System_Power (W)'] = results_df['Corrected_Avg_System_Power (W)'].map('{:.1f}'.format)
        results_df['Energy (J)'] = results_df['Energy (J)'].map('{:.1f}'.format)
        results_df['Energy_Uncertainty (J)'] = results_df['Energy_Uncertainty (J)'].map('{:.1f}'.format)
        results_df['Ratio Vs Max (Power)'] = (results_df['Ratio Vs Max (Power)'] * 100).map('{:.1f}%'.format)
        results_df['Ratio Vs Max (Power) Uncertainty'] = (results_df['Ratio Vs Max (Power) Uncertainty'] * 100).map('±{:.1f}%'.format)
        results_df['Ratio Vs Max (Energy)'] = (results_df['Ratio Vs Max (Energy)'] * 100).map('{:.1f}%'.format)
        results_df['Ratio Vs Max (Energy) Uncertainty'] = (results_df['Ratio Vs Max (Energy) Uncertainty'] * 100).map('±{:.1f}%'.format)

        print(results_df.to_string(index=False))
        print(f"\n{' ' * 20}Baseline Total System Power (W){' ' * 30}{baseline_power}\n")
        
        results_df.to_csv(output_csv, index=False)
        print(f"Results successfully saved to '{output_csv}'")
    else:
        print("\nNo valid inference sequences mapped. Check your directory structure.")
        
    return None

def main() -> None:
    """
    Main execution block handling CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze hardware telemetry and application logs for StrokeSeg2 benchmark.")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory containing hardware subfolders with CSVs and logs.")
    parser.add_argument("--baseline-power", type=float, default=32.6, help="Baseline system power (W) to subtract from total power.")
    parser.add_argument("--output-csv", type=str, default="inference_power_summary.csv", help="Filename for the exported CSV summary.")
    
    args = parser.parse_args()
    
    compute_energy_summary(
        root_dir=args.root_dir, 
        baseline_power=args.baseline_power,
        output_csv=args.output_csv
    )
    return None

if __name__ == "__main__":
    main()