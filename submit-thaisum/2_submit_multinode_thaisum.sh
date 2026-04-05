#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -t 20:00:00
#SBATCH -A ltxxxxxx
#SBATCH -J llamafac-thaisum
#SBATCH --output=../logs/train-thaisum.out

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export NCCL_TIMEOUT=3600000
export NCCL_BLOCKING_WAIT=0
export WANDB_MODE="offline"

export NNODES=$SLURM_NNODES
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 0-65535 -n 1)

echo "Nodes: $NNODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

srun bash multi_node_thaisum.sh
