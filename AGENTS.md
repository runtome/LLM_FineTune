# Repository Guidelines

## Project Structure & Module Organization
Top-level Python scripts drive the workflow: `prepare_asr_dataset.py` builds ShareGPT-style training data, `download_model.py` caches base weights, and `inference.py` / `inference_pretrained.py` run correction inference. Use `script/yaml/` for LLaMA Factory training configs, `script/deepspeed/` for DeepSpeed settings, and `script/dataset_info.config.json` for dataset registration. HPC submission scripts live in `submit-asr/` and `submit-script/`. Keep exploratory work in `notebook/`; treat `logs/`, `model/`, generated `dataset/`, and raw `datasets/` as runtime artifacts, not core source.

## Build, Test, and Development Commands
Set up the training environment with `bash setup.sh`; this creates `env-list/env`, clones LLaMA Factory, and installs DeepSpeed and notebook dependencies. Build formatted JSON training data with `python prepare_asr_dataset.py`. Download a base model for offline HPC runs with `python download_model.py --save_dir ./model/qwen3-4b`. Run inference locally with `python inference.py --model_path ./trained_model/qwen3-8b-asr`. On Lanta, submit jobs in sequence from `submit-asr/` with `sbatch 0_prepare-dataset.sh`, `sbatch 1_prepare-data-asr.sh`, `sbatch 2_submit_multinode_asr.sh`, and follow with merge and inference jobs.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, module-level constants in `UPPER_SNAKE_CASE`, functions and variables in `snake_case`, and concise docstrings on public helpers. Prefer small, script-oriented modules over deep package hierarchies. Match current file naming for configs and jobs: numeric prefixes such as `1_prepare-data-asr.sh` or `2_qwen3_lora_sft_asr.config.yaml` indicate pipeline order. No formatter or linter is checked in, so keep imports tidy and minimize unrelated rewrites.

## Testing Guidelines
There is no automated test suite in this repository today. Validate changes by running the affected script directly on a small sample and checking generated CSV or JSON output paths. For training-related edits, confirm that the referenced YAML, dataset paths, and `PROJECT_PATH` values in SLURM scripts are consistent before submitting jobs.

## Commit & Pull Request Guidelines
Git history is currently minimal (`Initial commit`), so use short imperative commit subjects such as `Add ASR dataset path validation`. Keep commits focused on one workflow step. PRs should describe the pipeline stage affected, list any changed scripts or YAML files, note required environment or path changes, and include representative command lines or log snippets when behavior changes.

## Security & Configuration Tips
Do not commit model weights, cached Hugging Face downloads, generated datasets, or cluster-specific secrets. Keep `PROJECT_PATH`, account codes, reservation names, and tokens in local or HPC-only configuration, and double-check paths before launching `sbatch` jobs.
