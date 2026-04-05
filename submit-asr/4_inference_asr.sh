#!/bin/bash
#SBATCH -p gpu                              # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1                                # Specify number of nodes
#SBATCH --ntasks-per-node=1                 # Specify number of tasks per node
#SBATCH --gpus-per-task=1                   # Specify number of GPUs per task
#SBATCH -t 10:00:00                         # Specify maximum time limit
#SBATCH -A zz991010                         # Specify project name
#SBATCH -J inference_asr                    # Specify job name
#SBATCH --output=../logs/inference-asr.out  # Output file

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

# Run inference on test set
python "$PROJECT_PATH/inference.py" \
    --model_path "$PROJECT_PATH/trained_model/qwen3-8b-asr" \
    --input_csv "$PROJECT_PATH/datasets/test/asr_ouput_test.csv" \
    --output_csv "$PROJECT_PATH/datasets/test/test_LLM_inferance.csv" \
    --max_new_tokens 512
