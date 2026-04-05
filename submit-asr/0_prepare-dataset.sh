#!/bin/bash
#SBATCH -p compute                              # Compute partition (no GPU, cheapest ~0.0078 SHr/core)
#SBATCH -N 1                                    # Specify number of nodes
#SBATCH -c 1                                    # 1 CPU core is enough for CSV processing
#SBATCH --ntasks-per-node=1                     # Specify tasks per node
#SBATCH -t 00:10:00                             # 10 minutes is more than enough
#SBATCH -A zz991010                             # Specify project name
#SBATCH -J prepare-asr-dataset                  # Specify job name
#SBATCH --output=../logs/prepare-asr-dataset.out # Output file

export PROJECT_PATH="/project/zz991000-zdeva/zz991010/MeetingTranscription/LLM/"

echo "User: $(whoami)"
echo "Hostname: $(hostname)"

ml purge
ml Mamba

conda deactivate
conda activate /project/zz991000-zdeva/zz991010/llamafactory

# Build ShareGPT JSON from paired CSVs
python "$PROJECT_PATH/prepare_asr_dataset.py" \
    --data_dir "$PROJECT_PATH/datasets" \
    --output_dir "$PROJECT_PATH/dataset/format_dataset/asr_correction"
