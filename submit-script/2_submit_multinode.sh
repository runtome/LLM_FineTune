#!/bin/bash
#SBATCH -p gpu                               # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1                                # Specify number of nodes
#SBATCH -c 64                               # Specify processors per task
#SBATCH --ntasks-per-node=1                 # Specify number of tasks per node
#SBATCH --gpus-per-node=4                   # Specify total number of GPUs per node
#SBATCH -t 1:00:00                          # Specify maximum time limit (72 hours)
#SBATCH -A ltxxxxxx                     # Specify project name
#SBATCH -J llamafac                         # Specify job name
#SBATCH --output=../logs/train.out           # Output file


# Environment setup
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export NCCL_TIMEOUT=3600000
export NCCL_BLOCKING_WAIT=0
export WANDB_MODE="offline"

# Distributed training setup
export NNODES=$SLURM_NNODES
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 0-65535 -n 1)

echo "Nodes: $NNODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Run the training script on all nodes
srun bash multi_node.sh
