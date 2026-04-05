#!/bin/bash
#SBATCH -p gpu                              # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1                                # Specify number of nodes
#SBATCH --ntasks-per-node=1                 # Specify number of tasks per node
#SBATCH --gpus-per-task=1                   # Specify number of GPUs per task
#SBATCH -t 30:00:00                         # Specify maximum time limit
#SBATCH -A zz991010                         # Specify project name
#SBATCH -J inference_pretrained             # Specify job name
#SBATCH --output=../logs/inference-pretrained.out  # Output file

export PROJECT_PATH="/project/zz991000-zdeva/zz991010/MeetingTranscription/LLM/" #YOUR PROJECT PATH

ml purge
ml cuda
ml gcc
ml Mamba

conda deactivate
conda activate /project/zz991000-zdeva/zz991010/llamafactory

# Change this path to your own path
export LD_LIBRARY_PATH=$PROJECT_PATH/lib:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=$PROJECT_PATH/.cache
export HF_DATASETS_CACHE=$PROJECT_PATH/.cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Run inference on test set using base pretrained model from HF cache
# Change --model_name to switch models (e.g. Qwen/Qwen2.5-7B, Qwen/Qwen3-8B)
python "$PROJECT_PATH/inference_pretrained.py" \
    --model_name "Qwen/Qwen3-8B" \
    --cache_dir "$PROJECT_PATH/.cache" \
    --input_csv "$PROJECT_PATH/datasets/test/asr_ouput_test.csv" \
    --output_csv "$PROJECT_PATH/datasets/test/test_pretrained_inference.csv" \
    --max_new_tokens 512
