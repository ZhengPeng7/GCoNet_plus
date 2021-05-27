#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00


# Run python script
sh ./run.sh


nvidia-smi
hostname
