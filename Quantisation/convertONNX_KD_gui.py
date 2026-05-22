import os
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from pathlib import Path
import torch.nn as nn

# nnU-Net Imports
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from batchgenerators.utilities.file_and_folder_operations import load_json

# dynamic_network_architectures Import (Required for custom KD models)
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

# ONNX Imports
import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType

# =========================================================================
# CUSTOM KD STUDENT ARCHITECTURES
# =========================================================================
from models import *
# =========================================================================
# EXPORTER APPLICATION
# =========================================================================

class nnUNetExporter:
    def __init__(self, root):
        self.root = root
        self.root.title("nnU-Net to ONNX Converter (KD Supported)")
        # Increased window height to accommodate the new Architecture field
        self.root.geometry("600x650")

        # Variables
        self.pth_path = tk.StringVar()
        self.plans_path = tk.StringVar()
        self.dataset_path = tk.StringVar()
        self.save_path = tk.StringVar()
        
        # Quantization list options
        self.quant_options = ["FP32 (No Quantization)", "FP16 (Recommended for GPU)", "INT8 Dynamic (Recommended for CPU/NPU)"]
        self.selected_quant = tk.StringVar(value=self.quant_options[0])

        # Architecture list options
        self.arch_options = [
            "Standard nnU-Net (From Plans)", 
            "KD Student - Large", 
            "KD Student - Medium", 
            "KD Student - Small", 
            "KD Student - Light",
            "KD Student - Extra Light",
            "KD Student - Extra Extra Light",
            "KD Student - Nano",
            "KD Student - Pico",
            "KD Student - Femto"
        ]
        self.selected_arch = tk.StringVar(value=self.arch_options[0])

        self.create_widgets()

    def create_widgets(self):
        pad = {'padx': 15, 'pady': 5}

        # 1. Architecture Selection (New)
        tk.Label(self.root, text="1. Select Model Architecture:", font=("Arial", 10, "bold")).pack(anchor="w", **pad)
        arch_menu = tk.OptionMenu(self.root, self.selected_arch, *self.arch_options)
        arch_menu.pack(fill="x", padx=15)

        # 2. Model Selection
        tk.Label(self.root, text="2. Select Checkpoint (.pth):", font=("Arial", 10, "bold")).pack(anchor="w", **pad)
        frame_pth = tk.Frame(self.root)
        frame_pth.pack(fill="x", padx=15)
        tk.Entry(frame_pth, textvariable=self.pth_path, width=50).pack(side="left", expand=True, fill="x")
        tk.Button(frame_pth, text="Browse", command=self.browse_pth).pack(side="right", padx=5)

        # 3. Plans Selection
        tk.Label(self.root, text="3. Select Plans File (.json):", font=("Arial", 10, "bold")).pack(anchor="w", **pad)
        frame_plans = tk.Frame(self.root)
        frame_plans.pack(fill="x", padx=15)
        tk.Entry(frame_plans, textvariable=self.plans_path, width=50).pack(side="left", expand=True, fill="x")
        tk.Button(frame_plans, text="Browse", command=self.browse_plans).pack(side="right", padx=5)

        # 4. Dataset Selection
        tk.Label(self.root, text="4. Select Dataset File (.json):", font=("Arial", 10, "bold")).pack(anchor="w", **pad)
        frame_dataset = tk.Frame(self.root)
        frame_dataset.pack(fill="x", padx=15)
        tk.Entry(frame_dataset, textvariable=self.dataset_path, width=50).pack(side="left", expand=True, fill="x")
        tk.Button(frame_dataset, text="Browse", command=self.browse_dataset).pack(side="right", padx=5)

        # 5. Save Selection
        tk.Label(self.root, text="5. Save ONNX Output As:", font=("Arial", 10, "bold")).pack(anchor="w", **pad)
        frame_save = tk.Frame(self.root)
        frame_save.pack(fill="x", padx=15)
        tk.Entry(frame_save, textvariable=self.save_path, width=50).pack(side="left", expand=True, fill="x")
        tk.Button(frame_save, text="Browse", command=self.browse_save).pack(side="right", padx=5)

        # 6. Quantization Selection
        tk.Label(self.root, text="6. Select Quantization Method:", font=("Arial", 10, "bold")).pack(anchor="w", **pad)
        quant_menu = tk.OptionMenu(self.root, self.selected_quant, *self.quant_options)
        quant_menu.pack(fill="x", padx=15)

        # 7. Action Button
        self.btn = tk.Button(self.root, text="CONVERT TO ONNX", bg="#2196F3", fg="white", 
                             font=("Arial", 12, "bold"), command=self.convert)
        self.btn.pack(pady=25, padx=15, fill="x")

    def browse_pth(self):
        file = filedialog.askopenfilename(filetypes=[("Weights", "*.pth")])
        if file:
            self.pth_path.set(file)
            self.save_path.set(file.replace(".pth", ".onnx"))
            
            try:
                trainer_dir = Path(file).parent.parent
                dataset_dir = trainer_dir.parent
                
                plans_file = next(trainer_dir.glob("*Plans*.json"), None)
                if plans_file:
                    self.plans_path.set(str(plans_file))
                
                dataset_file = dataset_dir / "dataset.json"
                if dataset_file.exists():
                    self.dataset_path.set(str(dataset_file))
            except Exception:
                pass

    def browse_plans(self):
        file = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file:
            self.plans_path.set(file)

    def browse_dataset(self):
        file = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file:
            self.dataset_path.set(file)

    def browse_save(self):
        file = filedialog.asksaveasfilename(defaultextension=".onnx", filetypes=[("ONNX", "*.onnx")])
        if file:
            self.save_path.set(file)

    def get_nnunet_model(self, pth_file, plans_file, dataset_file, arch_type):
        if not plans_file or not os.path.exists(plans_file):
            raise FileNotFoundError("Please provide a valid path to the plans.json file.")
        
        if not dataset_file or not os.path.exists(dataset_file):
            raise FileNotFoundError("Please provide a valid path to the dataset.json file.")

        plans = load_json(str(plans_file))
        dataset_json = load_json(str(dataset_file))
        
        num_input_channels = len(dataset_json["channel_names"])
        num_output_channels = len(dataset_json["labels"])        
        
        # Dynamically load the correct network based on the dropdown selection
        if arch_type == "Standard nnU-Net (From Plans)":
            config = plans["configurations"]["3d_fullres"]
            network = get_network_from_plans(
                config['architecture']['network_class_name'],
                config['architecture']['arch_kwargs'],
                config['architecture']['_kw_requires_import'],
                num_input_channels,
                num_output_channels,
                allow_init=True,
                deep_supervision=False 
            )
        elif arch_type == "KD Student - Large":
            network = get_large_student()
        elif arch_type == "KD Student - Medium":
            network = get_medium_student()
        elif arch_type == "KD Student - Small":
            network = get_small_student()
        elif arch_type == "KD Student - Light":
            network = get_light_student()
        elif arch_type == "KD Student - Extra Light":
            network = get_extra_light_student()
        elif arch_type == "KD Student - Extra Extra Light":
            network = get_extra_extralight_student()
        elif arch_type == "KD Student - Nano":
            network = get_nano_student()
        elif arch_type == "KD Student - Pico":
            network = get_pico_student()
        elif arch_type == "KD Student - Femto":
            network = get_femto_student()
        else:
            raise ValueError(f"Unknown architecture selected: {arch_type}")

        checkpoint = torch.load(pth_file, map_location='cpu', weights_only=False)
        network.load_state_dict(checkpoint['network_weights'])
        network.eval()
        return network, num_input_channels

    def convert(self):
        try:
            pth_file = self.pth_path.get()
            plans_file = self.plans_path.get()
            dataset_file = self.dataset_path.get()
            out_file = self.save_path.get()
            arch_type = self.selected_arch.get()

            if not pth_file or not plans_file or not dataset_file or not out_file:
                raise ValueError("Please ensure the Checkpoint, Plans, Dataset, and Output paths are all filled.")

            self.btn.config(text="CONVERTING...", state="disabled", bg="#9E9E9E")
            self.root.update()

            # Pass the selected architecture to the builder
            model, channels = self.get_nnunet_model(pth_file, plans_file, dataset_file, arch_type)
            dummy_input = torch.randn(1, channels, 128, 128, 128)
            
            tmp_fp32 = out_file if "No Quantization" in self.selected_quant.get() else "temp_fp32.onnx"
            
            torch.onnx.export(model, dummy_input, tmp_fp32, opset_version=18, 
                              input_names=['input'], output_names=['output'],
                              dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})

            choice = self.selected_quant.get()
            if "FP16" in choice:
                m = onnx.load(tmp_fp32)
                m_half = float16.convert_float_to_float16(m)
                onnx.save(m_half, out_file)
                os.remove(tmp_fp32)
            elif "INT8" in choice:
                quantize_dynamic(tmp_fp32, out_file, weight_type=QuantType.QInt8)
                os.remove(tmp_fp32)

            messagebox.showinfo("Success", f"Exported successfully to {out_file}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"Error during conversion: {e}")
        finally:
            self.btn.config(text="CONVERT TO ONNX", state="normal", bg="#2196F3")

if __name__ == "__main__":
    root = tk.Tk()
    app = nnUNetExporter(root)
    root.mainloop()