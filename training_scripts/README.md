# Training with HuggingFace Datasets

This guide explains how to upload your branch-generated trajectories to HuggingFace Hub and train models using the uploaded dataset.

## Environment setup

This repo’s training scripts expect **Python 3.10+** and a working CUDA toolchain (if training on GPU).

### 1) Create and activate a conda environment

```bash
conda create -p anchor_train python=3.10
conda activate ./anchor_train
```

### 2) Install PyTorch (CUDA 12.1)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3) Install training dependencies

```bash
pip install \
  "transformers>=4.44" \
  "accelerate>=0.33" \
  "datasets>=2.19" \
  deepspeed \
  pillow \
  tqdm
```

## Usage

### Basic Usage

```bash
# Train on all data (Windows + Ubuntu)
torchrun --nproc_per_node=4 train_qwenvl.py \
  --use_hf_dataset \
  --hf_dataset_name "mikeweii/ANCHOR" \
  --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
  --model_type "qwen2_5_vl" \
  --output_dir "output/qwen2.5-vl-7b-anchor"\
```

### Filter by OS

```bash
# Train only on Windows data
torchrun --nproc_per_node=4 train_qwenvl.py \
  --use_hf_dataset \
  --hf_dataset_name "mikeweii/ANCHOR" \
  --hf_dataset_os_filter "windows" \
  --model_name_or_path "Qwen/Qwen3-VL-8B-Instruct" \
  --model_type "qwen3vl" \
  --output_dir "output/qwen3vl_windows"

# Train only on Ubuntu data
torchrun --nproc_per_node=4 train_qwenvl.py \
  --use_hf_dataset \
  --hf_dataset_name "mikeweii/ANCHOR" \
  --hf_dataset_os_filter "ubuntu" \
  --model_name_or_path "Qwen/Qwen3-VL-8B-Instruct" \
  --model_type "qwen3vl" \
  --output_dir "output/qwen3vl_ubuntu"
```

### Train GLM-4V

```bash
torchrun --nproc_per_node=4 train_glm41v.py \
  --use_hf_dataset \
  --hf_dataset_name "mikeweii/branch-trajectories" \
  --model_name_or_path "THUDM/glm-4v-9b" \
  --output_dir "output/glm4v_all_os"
```

Notes:
- Set `--nproc_per_node` to your GPU count (e.g., `1`, `2`, `4`).
- If you’re launching via Slurm, `torchrun --standalone` is usually fine for single-node jobs.

## Key Arguments

### HuggingFace Dataset Arguments

- `--use_hf_dataset`: Enable loading from HuggingFace Hub
- `--hf_dataset_name`: Dataset name (e.g., `"username/dataset-name"`)
- `--hf_dataset_os_filter`: Filter by OS (`"windows"` or `"ubuntu"`, or omit for all)
- `--hf_cache_dir`: Custom cache directory for HuggingFace datasets

### Training Arguments

All existing training arguments are still supported:
- `--batch_size`: Effective batch size across all GPUs
- `--num_train_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--dual_training_types`: Enable both Type 1 and Type 2 training (default: True)
- `--max_past_screenshots`: Number of past screenshots to include (default: 2)
- `--save_strategy`: Save strategy (`"steps"` or `"epoch"`)
- `--save_steps`: Save checkpoint every N steps