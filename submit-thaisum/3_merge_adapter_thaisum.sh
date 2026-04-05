#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -t 10:00:00
#SBATCH -A ltxxxxxx
#SBATCH -J merge_adapter_thaisum
#SBATCH --output=../logs/merge-thaisum.out

export PROJECT_PATH="" #YOUR PROJECT PATH

ml purge
ml cuda
ml gcc
ml Mamba

conda deactivate
conda activate "$PROJECT_PATH/env-list/env"

export LD_LIBRARY_PATH=$PROJECT_PATH/lib:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=$PROJECT_PATH/.cache
export HF_DATASETS_CACHE=$PROJECT_PATH/.cache

envsubst < "$PROJECT_PATH/script/yaml/3_merge_lora_qwen3_thaisum.config.yaml" \
         > "$PROJECT_PATH/script/yaml/3_merge_lora_qwen3_thaisum.yaml"

echo "Resolved YAML:"
sed -n '1,20p' "$PROJECT_PATH/script/yaml/3_merge_lora_qwen3_thaisum.yaml"

llamafactory-cli export "$PROJECT_PATH/script/yaml/3_merge_lora_qwen3_thaisum.yaml"
