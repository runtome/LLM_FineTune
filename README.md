# LLM Fine-Tuning Pipeline for ASR Correction

Fine-tunes a Qwen3-4B model via LoRA SFT (using [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)) to correct errors and hallucinations in Thai ASR (Whisper) output.

## Two-Stage ASR Pipeline

1. **Whisper ASR** — raw transcription from audio (may contain hallucinations)
2. **Fine-tuned LLM** — corrects the raw ASR output into clean transcription

## Project Structure

```
LLM/
├── prepare_asr_dataset.py          # Build ShareGPT training data from paired CSVs
├── inference.py                    # Run correction inference on test data
├── setup.sh                        # Environment setup (conda, LLaMA Factory, deps)
├── notebook.sh                     # SLURM script for Jupyter notebook server
│
├── datasets/                       # Raw CSV data
│   ├── train/
│   │   ├── asr_output.csv          # 16,328 rows — raw Whisper ASR output
│   │   └── train.csv               # 16,328 rows — ground truth transcriptions
│   ├── val/
│   │   ├── asr_output.csv          # 2,312 rows
│   │   └── val.csv                 # 2,312 rows
│   └── test/
│       └── asr_ouput_test.csv      # 3,000 rows — test ASR output (no ground truth)
│
├── dataset/
│   └── format_dataset/
│       └── asr_correction/         # Generated ShareGPT JSON (output of prepare_asr_dataset.py)
│           ├── asr_correction_train.json
│           └── asr_correction_val.json
│
├── script/
│   ├── dataset_info.config.json    # LLaMA Factory dataset registry
│   ├── deepspeed/                  # DeepSpeed stage 0/2/3 configs
│   └── yaml/                       # Training YAML configs
│       ├── 1_data-process-asr.config.yaml
│       ├── 2_qwen3_lora_sft_asr.config.yaml
│       └── 3_merge_lora_qwen3_asr.config.yaml
│
├── submit-script/                  # SLURM job scripts (Lanta HPC)
│   ├── 1_prepare-data-asr.sh       # Tokenize dataset
│   ├── 2_submit_multinode_asr.sh   # Launch multi-node LoRA SFT training
│   ├── multi_node_asr.sh           # Worker script for distributed training
│   ├── 3_merge_adapter_asr.sh      # Merge LoRA adapter into base model
│   └── 4_inference_asr.sh          # Run inference on test set
│
├── notebook/                       # Jupyter notebooks (exploration & evaluation)
│   ├── data-process_no_run.ipynb
│   ├── finetune.ipynb
│   ├── merge_model_adapter.ipynb
│   └── evaluate.ipynb
│
├── model/                          # Base model weights (Qwen3-4B, not tracked in git)
└── logs/                           # SLURM job logs
```

## Setup

```bash
# On Lanta HPC — creates conda env and installs dependencies
bash setup.sh
```

**Requirements**: CUDA GPU, conda/Mamba, LLaMA Factory, DeepSpeed, transformers, torch

## Usage

### 1. Prepare Dataset

Pairs raw ASR output with ground truth transcriptions and converts to ShareGPT format:

```bash
python prepare_asr_dataset.py
```

Output: `dataset/format_dataset/asr_correction/asr_correction_{train,val}.json`

### 2. Train on Lanta HPC

Set `PROJECT_PATH` in each submit script, then submit SLURM jobs in order:

```bash
cd submit-script

# Step 1: Tokenize the dataset
sbatch 1_prepare-data-asr.sh

# Step 2: LoRA SFT training (multi-node, 4 GPUs)
sbatch 2_submit_multinode_asr.sh

# Step 3: Merge LoRA adapter into base model
sbatch 3_merge_adapter_asr.sh

# Step 4: Run inference on test set
sbatch 4_inference_asr.sh
```

### 3. Inference

Correct raw ASR transcriptions using the fine-tuned model:

```bash
# On Lanta HPC
sbatch 4_inference_asr.sh

# Or run directly
python inference.py \
  --model_path ./trained_model/qwen3-8b-asr \
  --input_csv  ./datasets/test/asr_ouput_test.csv \
  --output_csv ./datasets/test/test_LLM_inferance.csv
```

**CLI arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | (required) | Path to merged fine-tuned model |
| `--input_csv` | `datasets/test/asr_ouput_test.csv` | Input CSV (columns: path, sentence) |
| `--output_csv` | `datasets/test/test_LLM_inferance.csv` | Output CSV with corrected_sentence column |
| `--max_new_tokens` | 512 | Max generation length |
| `--batch_size` | 1 | Batch size (sequential processing) |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3-4B |
| Method | LoRA (rank 8, all linear layers) |
| DeepSpeed | Stage 2 |
| Max samples | 20,000 |
| Cutoff length | 4,096 tokens |
| Batch size | 8 per device |
| Learning rate | 1e-4 |
| Epochs | 1 |
| Precision | bf16 |

## Wisesight Sentiment (Demo Task)

The pipeline also includes a wisesight sentiment classification demo (the original workshop task). Its configs use the `wisesight_sentiment` dataset entry and the non-`asr` YAML/submit-script files.
