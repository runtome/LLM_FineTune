#!/bin/bash

ml purge
ml cuda
ml gcc
ml Mamba

export PROJECT_PATH="" #YOUR PROJECT PATH

conda deactivate
conda activate "$PROJECT_PATH/env-list/env"

echo "User: $(whoami)"
echo "Hostname: $(hostname)"
echo "SLURM_PROCID: $SLURM_PROCID"

# Change this path to your own path
export LD_LIBRARY_PATH=$PROJECT_PATH/lib:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=$PROJECT_PATH/cache
export HF_DATASETS_CACHE=$PROJECT_PATH/cache


# LLaMA Factory specific environment variables
export FORCE_TORCHRUN=1
export RANK=$SLURM_PROCID

# ===== Resolve config files =====
envsubst < "$PROJECT_PATH/script/yaml/2_qwen3_lora_sft_ds2.config.yaml" \
         > "$PROJECT_PATH/script/yaml/2_qwen3_lora_sft_ds2.yaml"

echo "Resolved YAML:"
sed -n '1,20p' "$PROJECT_PATH/script/yaml/2_qwen3_lora_sft_ds2.yaml"

# Run LLaMA Factory CLI
llamafactory-cli train "$PROJECT_PATH/script/yaml/2_qwen3_lora_sft_ds2.yaml"
