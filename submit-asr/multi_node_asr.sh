#!/bin/bash

ml purge
ml cuda
ml gcc
ml Mamba

export PROJECT_PATH="/project/zz991000-zdeva/zz991010/MeetingTranscription/LLM/" #YOUR PROJECT PATH

conda deactivate
conda activate /project/zz991000-zdeva/zz991010/llamafactory

echo "User: $(whoami)"
echo "Hostname: $(hostname)"
echo "SLURM_PROCID: $SLURM_PROCID"

# Change this path to your own path
export LD_LIBRARY_PATH=$PROJECT_PATH/lib:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=$PROJECT_PATH/.cache
export HF_DATASETS_CACHE=$PROJECT_PATH/.cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DISABLE_VERSION_CHECK=1


# LLaMA Factory specific environment variables
export FORCE_TORCHRUN=1
export RANK=$SLURM_PROCID

# ===== Resolve config files =====
envsubst < "$PROJECT_PATH/script/yaml/2_qwen3_lora_sft_asr.config.yaml" \
         > "$PROJECT_PATH/script/yaml/2_qwen3_lora_sft_asr.yaml"

echo "Resolved YAML:"
sed -n '1,20p' "$PROJECT_PATH/script/yaml/2_qwen3_lora_sft_asr.yaml"

# Run LLaMA Factory CLI
llamafactory-cli train "$PROJECT_PATH/script/yaml/2_qwen3_lora_sft_asr.yaml"
