# StrokeSeg-Toolkit

A modular toolkit for reproducing model distillation, quantization, evaluation, and runtime/energy analyses reported in the paper **"StrokeSeg2: Stroke Lesion Segmentation in Clinical Research Workflows"**.

> [!IMPORTANT]
> This repository contains **research/reproduction scripts**, not the end-user clinical software application.
> 
> The StrokeSeg2 software documentation is available at: https://strokeseg.readthedocs.io/

## Repository Structure

The codebase has been refactored into a clean monorepo architecture with four distinct modules:

```text
StrokeSeg-Toolkit/
├── Benchmark/         # Hardware inference orchestration and energy analysis
├── Distiller/         # ATLAS data preparation and KD student training scripts
├── Evaluation/        # Post-processing, metric calculations (Dice, F1, ASD), and plots
├── Exporter/          # Unified GUI for ONNX conversion and dynamic quantization
├── pyproject.toml     # Project configuration and dependency management
└── README.md
```

## Installation

This project is packaged using standard Python tools. It is recommended to install it in editable mode within your virtual environment (e.g., conda or venv) alongside your working `nnUNetv2` installation.

```bash
# Clone the repository
git clone https://github.com/Empenn-Stroke/StrokeSeg-Toolkit.git
cd StrokeSeg-Toolkit

# Install the toolkit and its dependencies in editable mode
pip install -e .
```

Installing via `pip install -e .` automatically sets up the required dependencies and registers command-line entry points for the GUI and benchmark tools.

## Reproduction Workflow

### 1. Data Preparation and Training (`Distiller`)
Prepare your ATLAS datasets and train Knowledge Distillation (KD) student models using the provided nnU-Net wrappers.
* **Prepare Training Data:**
  ```bash
  python Distiller/prepare_data.py --base-dir /path/to/ATLAS_2 --dataset-id 999
  ```
* **Prepare Test Split:**
  ```bash
  python Distiller/prepare_test.py --root-dir /path/to/ATLAS_2.1 --out-dir Test_Set_ATLAS_2.1
  ```
* **Train Student Models:**
  ```bash
  python Distiller/teach.py --dataset-id 999 --configuration 3d_fullres --fold 2
  ```

### 2. Model Export and Quantization (`Exporter`)
Convert PyTorch checkpoints to ONNX format and apply optional FP16 or INT8 dynamic quantization.
* **Launch the Unified GUI:**
  ```bash
  strokeseg-exporter
  ```
  *(This command is available globally once the package is installed via pip).*

### 3. Model Evaluation (`Evaluation`)
Compute clinical metrics (Global Dice, Lesion Dice, Lesion F1, ASD) and perform statistical equivalence tests.
* **Evaluate Predictions:**
  ```bash
  python Evaluation/evaluate_models.py --pred-dir preds/teacher --gt-dir Test_Set_ATLAS_2.1/masks --out-dir evaluation_results
  ```
* **Test FP32 vs FP16 Equivalence (TOST):**
  ```bash
  python Evaluation/test_equivalence.py --gt-dir Test_Set_ATLAS_2.1/masks --preds-base-dir preds
  ```
* **Generate Statistical Plots:**
  ```bash
  python Evaluation/plot_metrics.py
  ```

### 4. Hardware Benchmarking (`Benchmark`)
Run automated inference loops on the StrokeSeg2 executable and analyze energy consumption.
* **Run Batch Inference:**
  ```bash
  python Benchmark/run_benchmark.py --exe-path /path/to/strokeseg2-app.exe --input-dir /path/to/nifti/files
  ```
* **Analyze Power and Energy Logs:**
  ```bash
  strokeseg-benchmark --root-dir /path/to/hw/logs --baseline-power 32.6
  ```

## Citation and Acknowledgement

If you use this code, please cite:
- **StrokeSeg2: Stroke Lesion Segmentation in Clinical Research Workflows**

For the clinical software application (separate from this repository), refer to:
- https://strokeseg.readthedocs.io/