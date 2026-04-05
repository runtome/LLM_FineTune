# Thai Summarization Fine-Tuning Pipeline

This repository now supports a full bootstrap pipeline for **Thai text summarization** using the Hugging Face dataset [`pythainlp/thaisum`](https://huggingface.co/datasets/pythainlp/thaisum). The goal is to validate the end-to-end workflow first, then later swap in Thai meeting transcript data when you have your own dataset.

The current implementation uses:

- `load_dataset("pythainlp/thaisum")` to download the dataset
- only the `body` and `summary` columns for training
- Qwen3 + LoRA SFT with LLaMA Factory
- local JSONL exports under `dataset/`
- SLURM scripts for Lanta/HPC execution
- a manual Jupyter notebook for post-training inference

## 1. Repository Structure

```text
LLM_FineTune/
├── prepare_thaisum_dataset.py          # Download HF dataset and convert to local/ShareGPT files
├── summarization_inference.py          # Batch inference for summarization
├── evaluate_summarization.py           # ROUGE evaluation script
├── download_model.py                   # Optional offline base-model download helper
├── setup.sh                            # Conda + LLaMA Factory environment setup
├── notebook/
│   ├── manual_inference_thaisum.ipynb  # Manual inference after training
│   ├── finetune.ipynb
│   ├── evaluate.ipynb
│   └── merge_model_adapter.ipynb
├── dataset/
│   ├── thaisum_hf/                     # Cleaned local copy of ThaiSum
│   ├── format_dataset/
│   │   └── thai_news_summary/          # ShareGPT JSONL for LLaMA Factory
│   └── tokenized_dataset/
├── script/
│   ├── dataset_info.config.json
│   ├── deepspeed/
│   └── yaml/
│       ├── 1_data-process-thaisum.config.yaml
│       ├── 2_qwen3_lora_sft_thaisum.config.yaml
│       └── 3_merge_lora_qwen3_thaisum.config.yaml
├── submit-thaisum/
│   ├── 0_prepare-thaisum-dataset.sh
│   ├── 1_prepare-data-thaisum.sh
│   ├── 2_submit_multinode_thaisum.sh
│   ├── 3_merge_adapter_thaisum.sh
│   ├── 4_inference_thaisum.sh
│   └── multi_node_thaisum.sh
└── logs/
```

## 2. Dataset Bootstrap: Download ThaiSum into `dataset/`

The first step is to download the dataset from Hugging Face and save it locally.

Core loading method:

```python
from datasets import load_dataset

ds = load_dataset("pythainlp/thaisum")
```

This dataset includes several columns such as `title`, `body`, `summary`, `type`, `tags`, and `url`. For summarization training, this pipeline keeps only:

- `body`: source article text
- `summary`: reference summary

The script `prepare_thaisum_dataset.py` does all of the following:

1. downloads ThaiSum
2. removes unused columns
3. filters empty `body` or `summary`
4. saves the cleaned Hugging Face dataset to `dataset/thaisum_hf`
5. exports per-split JSONL files
6. converts each split into ShareGPT JSONL for LLaMA Factory

Run locally:

```bash
python prepare_thaisum_dataset.py
```

Expected outputs:

```text
dataset/thaisum_hf/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── ...HF save_to_disk files...

dataset/format_dataset/thai_news_summary/
├── thai_news_summary_train.json
├── thai_news_summary_validation.json
└── thai_news_summary_test.json
```

## 3. Prompt Strategy for Fine-Tuning

The default supervised prompt is summarization-focused, not meeting-minutes-focused yet. Each sample is converted to ShareGPT format:

```json
{
  "messages": [
    {"role": "system", "content": "คุณเป็นผู้ช่วยสรุปข่าวภาษาไทย ..."},
    {"role": "user", "content": "<body>"},
    {"role": "assistant", "content": "<summary>"}
  ]
}
```

Recommended prompt design principles:

- keep the task narrow: summarize only from source text
- forbid hallucination: do not add facts not present in the article
- prefer concise Thai output
- avoid chain-of-thought in inference by adding `/no_think`

Recommended prompt variants for future experimentation:

1. Concise summary
2. Three-sentence summary
3. Key-points summary

For the first pipeline version, keep the target format simple: **one concise Thai summary paragraph**.

## 4. Model Selection and Fine-Tuning

Recommended starting model:

- `Qwen/Qwen3-4B` for cheaper, faster validation

Upgrade path:

- `Qwen/Qwen3-8B` if quality is not enough and GPU memory allows it

Current summarization configs:

- `script/yaml/1_data-process-thaisum.config.yaml`
- `script/yaml/2_qwen3_lora_sft_thaisum.config.yaml`
- `script/yaml/3_merge_lora_qwen3_thaisum.config.yaml`

Training settings currently default to:

- LoRA SFT
- cutoff length `4096`
- LoRA rank `16`
- batch size `4`
- gradient accumulation `4`
- epochs `3`

### Local Setup

```bash
bash setup.sh
```

This creates `env-list/env`, clones LLaMA Factory, and installs the extra packages needed for summarization preparation and evaluation.

### Training Workflow on HPC

Before running SLURM jobs, set `PROJECT_PATH` inside the scripts in `submit-thaisum/`.

Step 0, download and prepare dataset:

```bash
bash submit-thaisum/0_prepare-thaisum-dataset.sh
```

Important: this step needs internet access because it pulls from Hugging Face.

Step 1, tokenize:

```bash
sbatch submit-thaisum/1_prepare-data-thaisum.sh
```

Step 2, train LoRA adapter:

```bash
cd submit-thaisum
sbatch 2_submit_multinode_thaisum.sh
```

Step 3, merge adapter into the base model:

```bash
sbatch submit-thaisum/3_merge_adapter_thaisum.sh
```

Merged model output:

```text
trained_model/qwen3-4b-thaisum
```

## 5. Batch Inference

Use `summarization_inference.py` for batch summarization after training.

Example:

```bash
python summarization_inference.py \
  --model_path ./trained_model/qwen3-4b-thaisum \
  --input_path ./dataset/thaisum_hf/validation.jsonl \
  --output_path ./dataset/thaisum_hf/validation_predictions.csv \
  --text_column body \
  --reference_column summary \
  --max_new_tokens 256
```

Output columns:

- `body`
- `generated_summary`
- `summary` if the reference column exists

SLURM version:

```bash
sbatch submit-thaisum/4_inference_thaisum.sh
```

## 6. Evaluation Method

Use `evaluate_summarization.py` to compute ROUGE on validation or test predictions.

Example:

```bash
python evaluate_summarization.py \
  --prediction_path ./dataset/thaisum_hf/validation_predictions.csv \
  --prediction_column generated_summary \
  --reference_column summary \
  --output_path ./dataset/thaisum_hf/validation_metrics.json
```

Current metrics:

- `rouge1`
- `rouge2`
- `rougeL`

Recommended manual review checklist:

- factual consistency with the source body
- coverage of the main point
- over-compression
- repeated or malformed phrases
- hallucinated details

## 7. Manual Inference Notebook

Use `notebook/manual_inference_thaisum.ipynb` after training to:

- load the merged model
- paste a Thai article manually
- generate a summary interactively
- test prompt and generation settings without rerunning batch inference

Default notebook use case:

1. set `PROJECT_PATH`
2. set `MODEL_PATH`
3. paste Thai text into `input_text`
4. run the inference cell

## 8. How This Connects to Thai Meeting Minutes Later

ThaiSum is only the bootstrap dataset for validating the pipeline. Later, when you have meeting transcript data, you can reuse almost the entire system:

- replace `prepare_thaisum_dataset.py` with a meeting-data preparation script
- change the prompt from news summarization to meeting summarization
- change the target from article summary to executive summary or meeting minutes
- keep the same Qwen3 + LoRA + LLaMA Factory training path

For a future meeting dataset, the minimum useful schema is:

- `meeting_id`
- `speaker`
- `start_time`
- `end_time`
- `utterance`
- `reference_summary`

## 9. Useful Commands

Prepare dataset:

```bash
python prepare_thaisum_dataset.py
```

Train:

```bash
sbatch submit-thaisum/1_prepare-data-thaisum.sh
sbatch submit-thaisum/2_submit_multinode_thaisum.sh
sbatch submit-thaisum/3_merge_adapter_thaisum.sh
```

Inference:

```bash
python summarization_inference.py --model_path ./trained_model/qwen3-4b-thaisum
```

Evaluation:

```bash
python evaluate_summarization.py --prediction_path ./dataset/thaisum_hf/validation_predictions.csv
```

## 10. Notes and Limitations

- ThaiSum is news summarization, not meeting summarization.
- Good ROUGE on ThaiSum does not guarantee good meeting-minutes quality.
- Long meeting transcripts may require chunking or hierarchical summarization later.
- If Hugging Face access is blocked on HPC, run `prepare_thaisum_dataset.py` on a machine with internet first, then copy the `dataset/` artifacts to the cluster.
