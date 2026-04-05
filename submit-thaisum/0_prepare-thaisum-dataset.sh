#!/bin/bash
# Run on a machine or node with internet access because it downloads from Hugging Face.

export PROJECT_PATH="" #YOUR PROJECT PATH

ml purge
ml Mamba

conda deactivate
conda activate "$PROJECT_PATH/env-list/env"

python "$PROJECT_PATH/prepare_thaisum_dataset.py" \
    --raw_output_dir "$PROJECT_PATH/dataset/thaisum_hf" \
    --sharegpt_output_dir "$PROJECT_PATH/dataset/format_dataset/thai_news_summary"
