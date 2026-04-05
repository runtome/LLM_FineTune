#!/bin/bash
#SBATCH -p gpu				    # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16         		# Specify number of nodes and processors per task
#SBATCH --gpus=1                # Specify the number of GPUs
#SBATCH --ntasks-per-node=1		# Specify tasks per node
#SBATCH --mem=100G
#SBATCH -t 2:00:00			# Specify maximum time limit (hour: minute: second)
#SBATCH -A ltxxxxxx			# Specify project name
#SBATCH -J nb               # Specify job name
#SBATCH --output=./logs/nb.out
#SBATCH --reservation=thaisc_311

export PROJECT_PATH="" #YOUR PROJECT PATH

module restore
ml purge
ml cuda
ml gcc
ml Mamba

conda deactivate
conda activate "$PROJECT_PATH/env-list/env"

port=$(shuf -i 6000-9999 -n 1)
USER=$(whoami)
node=$(hostname -s)

#jupyter notebookng instructions to the output file
echo -e "

    Jupyter server is running on: $(hostname)
    Job starts at: $(date)

    Copy/Paste the following command into your local terminal 
    --------------------------------------------------------------------
    ssh -L $port:$node:$port $USER@lanta.nstda.or.th -i .\.ssh\id_rsa
    --------------------------------------------------------------------

    Open a browser on your local machine with the following address
    --------------------------------------------------------------------
    http://localhost:${port}/?token=XXXXXXXX (see your token below)
    --------------------------------------------------------------------    
    "

## start a cluster instance and launch jupyter server

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

export HF_HOME=$PROJECT_PATH/.cache
jupyter notebook --no-browser --port $port --notebook-dir=$(pwd) --ip=$node
