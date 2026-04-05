#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -t 10:00:00
#SBATCH -A ltxxxxxx
#SBATCH -J inference_thaisum
#SBATCH --output=../logs/inference-thaisum.out

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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python "$PROJECT_PATH/summarization_inference.py" \
    --model_path "$PROJECT_PATH/trained_model/qwen3-4b-thaisum" \
    --input_path "$PROJECT_PATH/dataset/thaisum_hf/validation.jsonl" \
    --output_path "$PROJECT_PATH/dataset/thaisum_hf/validation_predictions.csv" \
    --text_column body \
    --reference_column summary \
    --max_new_tokens 256
