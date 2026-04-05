#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 10:00:00
#SBATCH -A ltxxxxxx
#SBATCH -J pre-tokenizer-thaisum
#SBATCH --output=../logs/pre-tokenized-thaisum.out

export PROJECT_PATH="" #YOUR PROJECT PATH
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6"

export HF_DATASETS_CACHE="$PROJECT_PATH/.cache"
export HF_HOME="$PROJECT_PATH/.cache"
export HF_HUB_CACHE="$PROJECT_PATH/.cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export DISABLE_VERSION_CHECK=1

ml purge
ml cuda
ml gcc
ml Mamba

conda deactivate
conda activate "$PROJECT_PATH/env-list/env"

envsubst < "$PROJECT_PATH/script/yaml/1_data-process-thaisum.config.yaml" \
         > "$PROJECT_PATH/script/yaml/1_data-process-thaisum.yaml"

envsubst < "$PROJECT_PATH/script/dataset_info.config.json" \
         > "$PROJECT_PATH/script/dataset_info.json"

echo "Resolved YAML:"
sed -n '1,20p' "$PROJECT_PATH/script/yaml/1_data-process-thaisum.yaml"

echo "Resolved dataset_info.json:"
sed -n '1,30p' "$PROJECT_PATH/script/dataset_info.json"

llamafactory-cli train "$PROJECT_PATH/script/yaml/1_data-process-thaisum.yaml"
