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

export LD_LIBRARY_PATH=$PROJECT_PATH/lib:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=$PROJECT_PATH/.cache
export HF_DATASETS_CACHE=$PROJECT_PATH/.cache
export HF_HUB_CACHE=$PROJECT_PATH/.cache
export FORCE_TORCHRUN=1
export RANK=$SLURM_PROCID

envsubst < "$PROJECT_PATH/script/yaml/2_qwen3_lora_sft_thaisum.config.yaml" \
         > "$PROJECT_PATH/script/yaml/2_qwen3_lora_sft_thaisum.yaml"

echo "Resolved YAML:"
sed -n '1,25p' "$PROJECT_PATH/script/yaml/2_qwen3_lora_sft_thaisum.yaml"

llamafactory-cli train "$PROJECT_PATH/script/yaml/2_qwen3_lora_sft_thaisum.yaml"
