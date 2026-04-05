#!/bin/bash
#SBATCH -p gpu                              # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1                                # Specify number of nodes
#SBATCH --ntasks-per-node=1                 # Specify number of tasks per node
#SBATCH --gpus-per-task=1                   # Specify number of GPUs per task
#SBATCH -t 2:00:00                          # Specify maximum time limit (72 hours)
#SBATCH -A ltxxxxxx                     # Specify project name
#SBATCH -J merge_adapter                    # Specify job name
#SBATCH --output=../logs/merge.out           # Output file
#SBATCH --reservation=thaisc_311

export PROJECT_PATH="" #YOUR PROJECT PATH

ml purge
ml cuda
ml gcc
ml Mamba

conda deactivate
conda activate "$PROJECT_PATH/env-list/env"

# Change this path to your own path
export LD_LIBRARY_PATH=$PROJECT_PATH/lib:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=$PROJECT_PATH/.cache
export HF_DATASETS_CACHE=$PROJECT_PATH/.cache
export FORCE_TORCHRUN=1

# ===== Resolve config files =====
envsubst < "$PROJECT_PATH/script/yaml/3_merge_lora_qwen3_sft.config.yaml" \
         > "$PROJECT_PATH/script/yaml/3_merge_lora_qwen3_sft.yaml"

echo "Resolved YAML:"
sed -n '1,20p' "$PROJECT_PATH/script/yaml/3_merge_lora_qwen3_sft.config.yaml"

# Merge Adapter
llamafactory-cli export "$PROJECT_PATH/script/yaml/3_merge_lora_qwen3_sft.yaml"
