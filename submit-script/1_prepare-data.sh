#!/bin/bash
#SBATCH -p gpu                                  # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1                                    # Specify number of nodes
#SBATCH -c 32                                   # Specify processors per task
#SBATCH --gpus-per-node=1                       # Specify number of gpu
#SBATCH --ntasks-per-node=1                     # Specify tasks per node
#SBATCH -t 1:00:00                              # Specify job time limit
#SBATCH -J pre-tokenizer                        # Specify job name
#SBATCH -A ltxxxxxx                             # Specify project name
#SBATCH --output=../logs/pre-tokenized.out        # Output file
#SBATCH --reservation=thaisc_311

export PROJECT_PATH="" #YOUR PROJECT PATH
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6"

export HF_DATASETS_CACHE="$PROJECT_PATH/.cache"
export HF_HOME="$PROJECT_PATH/.cache"
export HF_HUB_CACHE="$PROJECT_PATH/.cache"


export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

export DISABLE_VERSION_CHECK=1

echo "User: `whoami`"
echo "Count Node: $COUNT_NODE"
echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
echo "PATH = $PATH"
echo "Which mpicc: `which mpicc`"
echo "Hostnames: $HOSTNAMES"
echo "Hostname: `hostname`"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Determine the rank of the current node
H=`hostname`
THEID=`echo -e $HOSTNAMES | python -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo "THEID (Node Rank): $THEID"
echo "SLURM_PROCID: $SLURM_PROCID"


ml purge
ml cuda
ml gcc
ml Mamba

conda deactivate
conda activate "$PROJECT_PATH/env-list/env"

# ===== Resolve config files =====
envsubst < "$PROJECT_PATH/script/yaml/1_data-process-qwen3.config.yaml" \
         > "$PROJECT_PATH/script/yaml/1_data-process-qwen3.yaml"

envsubst < "$PROJECT_PATH/script/dataset_info.config.json" \
         > "$PROJECT_PATH/script/dataset_info.json"

echo "Resolved YAML:"
sed -n '1,20p' "$PROJECT_PATH/script/yaml/1_data-process-qwen3.yaml"

echo "Resolved dataset_info.json:"
sed -n '1,20p' "$PROJECT_PATH/script/dataset_info.json"

# tokenize
llamafactory-cli train "$PROJECT_PATH/script/yaml/1_data-process-qwen3.yaml"
