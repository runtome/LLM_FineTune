#!/bin/bash
#SBATCH -p gpu                              # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1                                # Specify number of nodes
#SBATCH --ntasks-per-node=1                 # Specify number of tasks per node
#SBATCH --gpus-per-task=1                   # Specify number of GPUs per task
#SBATCH -t 10:00:00                          # Specify maximum time limit (72 hours)
#SBATCH -A zz991010			                    # Specify project name
#SBATCH -J merge_adapter_asr               # Specify job name
#SBATCH --output=../logs/merge-asr.out      # Output file

export PROJECT_PATH="/project/zz991000-zdeva/zz991010/MeetingTranscription/LLM/" #YOUR PROJECT PATH

ml purge
ml cuda
ml gcc
ml Mamba

conda deactivate
conda activate /project/zz991000-zdeva/zz991010/llamafactory

# Change this path to your own path
export LD_LIBRARY_PATH=$PROJECT_PATH/lib:$LD_LIBRARY_PATH
export HF_HOME=$PROJECT_PATH/.cache
export HF_HUB_CACHE=$PROJECT_PATH/.cache
export HF_DATASETS_CACHE=$PROJECT_PATH/.cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ===== Resolve config files =====
envsubst < "$PROJECT_PATH/script/yaml/3_merge_lora_qwen3_asr.config.yaml" \
         > "$PROJECT_PATH/script/yaml/3_merge_lora_qwen3_asr.yaml"

echo "Resolved YAML:"
sed -n '1,20p' "$PROJECT_PATH/script/yaml/3_merge_lora_qwen3_asr.yaml"

# Merge Adapter
llamafactory-cli export "$PROJECT_PATH/script/yaml/3_merge_lora_qwen3_asr.yaml"
