# StrokeSegDistillationQuantisation

Experimental code for reproducing model distillation, quantisation, evaluation, and runtime/energy analyses reported in the paper **"StrokeSeg2: Stroke Lesion Segmentation in Clinical Research Workflows"**.

> [!IMPORTANT]
> This repository contains **research/reproduction scripts**, not the end-user clinical software application.
> 
> The StrokeSeg2 software documentation is available at: https://strokeseg-doc.readthedocs.io/

## What this repository supports

This codebase is organized around the paper experiments:
- preparing ATLAS-style datasets for nnU-Net workflows,
- training teacher/student distillation variants,
- exporting and quantising trained checkpoints,
- running runtime inference experiments,
- summarizing accuracy/runtime/energy outcomes and significance.

## Repository structure

```text
.
├── Distillation/
│   ├── PrepareData.py                # Build nnU-Net Dataset999 from ATLAS-style data, align/binarize masks,
│   │                                 # write dataset.json, launch nnUNetv2 planning/preprocessing
│   ├── prepareTest.py                # Extract ATLAS v2.1 test split into nnU-Net-style image/mask files
│   ├── teach.py                      # Custom nnU-Net KD trainer (teacher+student, T=4.0, alpha=0.5,
│   │                                 # 1000 epochs, teacher checkpoint loading, frozen teacher)
│   ├── models.py                     # Student 3D ResidualEncoderUNet size variants (large -> femto/pico/nano)
│   ├── lesion_wise_evaluation.py     # Lesion-wise evaluation helper
│   ├── quantisation_equivalence.py   # Quantisation output consistency checks
│   ├── boxplot.py                    # Paired comparison plotting + Wilcoxon significance reporting
│   └── evaluation/                   # Additional plotting scripts for metric summaries
├── Quantisation/
│   └── convertONNXgui.py             # GUI export to ONNX + FP16 / INT8 dynamic quantisation options
└── RuntimeExp/
    ├── RuntimeExp.py                 # Batch runtime experiment driver (runs installed StrokeSeg app on NIfTI files)
    └── main.py                       # Parse app logs + hardware CSVs to compute inference time and energy summary
```

## Data and inputs

To reproduce the pipeline you will typically need:
- **ATLAS v2.0/v2.1 data** in an ATLAS-style folder layout.
- A working **nnU-Net v2** environment (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`).
- A pretrained **teacher checkpoint** (expected by `teach.py` under the nnU-Net results tree).
- Runtime experiment artifacts:
  - app inference logs (`.log`),
  - hardware monitoring CSV files (power/time columns),
  - NIfTI inputs for runtime runs.

## Reproduction workflow (high level)

1. **Prepare ATLAS training data for nnU-Net**
   - Adapt paths in `Distillation/PrepareData.py`.
   - Run:
     ```bash
     python /absolute/path/to/Distillation/PrepareData.py
     ```

2. **Prepare ATLAS v2.1 test split**
   - Adapt input/output paths in `Distillation/prepareTest.py`.
   - Run:
     ```bash
     python /absolute/path/to/Distillation/prepareTest.py
     ```

3. **Train distillation models**
   - Confirm environment variables and teacher checkpoint path in `Distillation/teach.py`.
   - Select the student trainer variant you want to run.
   - Run:
     ```bash
     python /absolute/path/to/Distillation/teach.py
     ```

4. **Export and quantise checkpoints**
   - Launch ONNX conversion GUI:
     ```bash
     python /absolute/path/to/Quantisation/convertONNXgui.py
     ```
   - Choose FP32 / FP16 / INT8 dynamic quantisation in the UI.

5. **Run runtime experiments**
   - Edit executable/input/output paths in `RuntimeExp/RuntimeExp.py`.
   - Run inference batch experiments for selected model variants.

6. **Compute runtime + energy summaries**
   - Organize logs/CSV files as expected by `RuntimeExp/main.py`.
   - Run:
     ```bash
     python /absolute/path/to/RuntimeExp/main.py
     ```
   - Output includes `inference_power_summary.csv`.

7. **Statistical comparison and plots**
   - Use `Distillation/boxplot.py` (and associated evaluation scripts) on model result CSVs to generate paired comparison plots and significance outputs.

## Environment/setup notes

- Scripts are primarily Python-based and depend on scientific imaging/ML libraries (for example: PyTorch, nnU-Net v2, nibabel, nilearn, pandas, scipy, seaborn, matplotlib, onnx, onnxruntime).
- This repository does not currently provide a fully pinned environment file; set up dependencies according to your nnU-Net and experiment platform.

## Path cautions

Several scripts contain **hard-coded absolute paths** (Linux and Windows examples). Before running experiments, update these paths for your machine:
- `Distillation/PrepareData.py`
- `Distillation/teach.py`
- `RuntimeExp/RuntimeExp.py`
- other helper scripts as needed

## Citation and acknowledgement

If you use this code, please cite:
- **StrokeSeg2: Stroke Lesion Segmentation in Clinical Research Workflows**

And for the software application (separate from this repository), refer to:
- https://strokeseg-doc.readthedocs.io/
