# Accelerating medical diagnostics algorithms on AMD GPUs

## Project Overview

This repository provides a hardware-adaptive PyTorch pipeline for binary image classification using a pretrained DenseNet-121 backbone plus an optional spatial (chunked) evaluation stage that generates interpretability heatmaps. It is optimized to run on both:
- AMD GPUs - with explicit ROCm tuning
- NVIDIA GPUs - with torch.compile, TF32 + bfloat16

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Hardware](#hardware)
- [Installation](#installation)
- [Dataset](#dataset)
- [Running](#running)
- [Chunked Evaluation](#chunked-evaluation)
- [Environment Variables](#environment-variables)
- [Optimization Summary](#optimization-summary)
- [License](#license)
- [Contributors](#contributors)

---

## Key Features

| Area | Feature |
|------|---------|
| Model | DenseNet-121 (ImageNet pretrained) → single-logit head |
| Loss | Weighted BCEWithLogitsLoss (handles class imbalance) |
| Sampler | WeightedRandomSampler to balance minority class exposure |
| Precision | Automatic Mixed Precision (bf16 on NVIDIA, fp16/bf16 on AMD) |
| Memory Format | Channels Last (NHWC) for better tensor core / MFMA utilization |
| Optimization (AMD) | ROCm env tuning, pinned memory, large BAR |
| Optimization (NVIDIA) | torch.compile, TF32 enabled, dynamic quantization (classifier) |
| Evaluation | Standard val loop + chunked inference with overlapping tiles (heatmaps) |
| Interpretability | Per-image probabilistic heatmaps & “bad” case extraction |
| Logging | JSON metadata (hardware, run config, accuracy, timings) |
| Artifacts | Best model checkpoint + optimized variant (TorchScript / quantized) |
| Robustness | OOM handling, safe fallbacks, optional AMP scaler tuning |

---

## Hardware

### System Configurations:

| System   | CPU                          | GPU                         | OS               | RAM                        | CUDA/ROCm  | Python / PyTorch |
|----------|------------------------------|-----------------------------|------------------|----------------------------|------------|------------------|
| System 1 | AMD Ryzen 5 7600X | NVIDIA RTX 4080 Super 16GB | Ubuntu 24.04 | 64 GB DDR5@6000MT/S | CUDA 12.4  | 3.11.13 / 2.7.1 |
| System 2 | AMD Ryzen 9 7950X | AMD RX 7900 XTX 24GB | Ubuntu 24.04 | 128 GB DDR4@3200MT/S | ROCm 6.4.1 | 3.13.5 / 2.7.1 |
| System 3 | Intel Core i5-13420H | NVIDIA RTX 4050 Laptop 6GB  | Arch Linux | 32 GB DDR5@5200MT/s | CUDA 13.0  | 3.13.7 / 2.8.0 |
| System 4 | AMD Ryzen 7 5800H | NVIDIA RTX 3050 Laptop 4GB | NixOS 25.05 | 24 GB DDR4@3200MT/s | CUDA 12.8 | 3.13.7 / 2.8.0 |


---

## Installation

1. Create environment
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies (choose appropriate torch wheel):
```bash
pip install -r requirements.txt

# For NVIDIA (replace --index-url with appropiate version for the used GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# For AMD ROCm (replace --index-url with appropiate version for the used GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3 
```

3. (Optional AMD) Ensure ROCm runtime libraries are installed and GPU is supported.

---

## Dataset

We extend our heartfelt appreciation to Dr. Emanuel Cojocariu from County Hospital Brasov for his exceptional contribution to this research. 


Dr. Cojocariu's specialized knowledge in cardiology proved essential in defining the clinical benchmarks that guided our algorithm development.

His generous sharing of expertise regarding coronary angiogram analysis and vessel identification greatly strengthened the clinical applicability of our methodology. We are especially thankful for his provision of the comprehensive coronary angiogram dataset that formed the foundation of this study.

---

## Running

Basic (for linux systems):
```bash
./dense.py
or for model demo
./gui.py`
```
Universal:
```bash
python3 dense.py
or for model demo
python3 gui.py
```

---

## Chunked Evaluation (Heatmaps)

The model is a global classifier. The chunking phase simulates localization:

    1. Split each test image into overlapping 64×64 tiles (stride = 48).
    2. Inference over all chunks (batched per image).
    3. Reconstruct a confidence heatmap by averaging overlapping areas.
    4. Derive:
    - Image-level prediction (mean chunk probability > 0.5)
    - Per-image mean confidence
    - “Bad” images → saved overlays.

Benefits:
- Spatial interpretability
- Error triage (focus on mispredictions or high-confidence positives)
- Threshold analysis across aggregated confidences

---

## Environment Variables

| Variable                    | Value    | Purpose                                                                                       |
|-----------------------------|----------|-----------------------------------------------------------------------------------------------|
| `HIP_PLATFORM`              | `amd`    | Force hip to use AMD instead of NVIDIA                                                        |                           |
| `HIP_LAUNCH_BLOCKING`       | `0`      | Enable async kernel, CPU continues while GPU works                                            |
| `ROCM_FORCE_CUDA_COMPAT`    | `1`      | Enables cuda compatibility layer                                                              | 
| `HSA_OVERRIDE_GFX_VERSION`  | `11.0.3` | Use stable version for RX 7900 XTX (testing gpu)                                              |
| `HIP_FORCE_DEV_KERNARG`     | `1`      | Force kernal args to be stored in ram, helps with frequent kernel calls                       |
| `HSA_FORCE_FINE_GRAIN_PCIE` | `1`      | Enable fine grain for pcie, for frequent small memory transfers                               |
| `HIP_HOST_COHERENT`         | `0`      | Double edged sword, disables host-coherent memory, requires cache management                  |
| `HSA_LARGE_BAR`             | `1`      | Enables large bar support, allows CPU to access GPU memory directly                           |
| `ROC_ENABLE_LARGE_BAR`      | `1`      | Same as above                                                                                 |
| `HSA_ENABLE_INTERRUPT`      | `0`      | Reduces per-kernel wakeup latency and smooths batch/iteration times                                    |

---

## Optimization Summary

| Aspect | AMD (ROCm) | NVIDIA |
|--------|------------|--------|
| Env tuning | HIP/ROCm vars (pinned memory, large BAR, async) | Standard |
| Compile | Disabled (stability reasons) | Enabled if available |
| Precision | fp16 AMP | fp16 AMP |
| Quantization (dynamic) | Skipped due to evaluation issues | Applied to final Linear head if scripted |
| Heatmaps | Same method | Same method |

---

## Loss & Optimization

- Loss: `BCEWithLogitsLoss(pos_weight=...)`
  - Combines sigmoid + weighted binary cross entropy
  - Mitigates class imbalance

- Optimizer: `Adam`
  - Adaptive learning rates (good for pretrained fine-tuning)
  - Could be swapped for `AdamW` or fused variants (NVIDIA) if desired

---

## Mixed Precision Logic

| Condition | Dtype Used |
|-----------|------------|
| NVIDIA | bfloat16 |
| AMD | fp16 |
| CPU | float32 |



---

## License

This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE.txt) file for details.

---

## Contributors

- Cătălina Bandas, Email: catalina.bandas@student.unitbv.ro
- Bogdan-Valentin Floricescu, Email: bogdan.floricescu@student.unitbv.ro
- George-Ștefan Ionescu, Email: stefan.ionescu@student.unitbv.ro
- Cristian-Nicolae Carcalețeanu, Email: cristian.carcaleteanu@student.unitbv.ro
- Vladuț-Gabriel Anghel, Email: vladut.anghel@student.unitbv.ro

Coordinator: Lecturer Cătălin Ciobanu, Email: catalin.ciobanu@unitbv.ro

---
For detailed information to the demonstration video, which is available here: [Demo Video](https://www.youtube.com/watch?v=JDg0VmjIc3c)
