#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00


# Run python script
method="gconet_$1"
size=300
epochs=65
val_last=10

# Train
python train.py --loss DSLoss_IoU_noCAM --trainset DUTS_class --size ${size} --ckpt_dir ckpt/${method} --lr 3e-4 --bs 48 --epochs 55

for ((ep=${epochs}-${val_last};ep<${epochs};ep++))
do
python test.py --pred_dir /home/pz1/datasets/sod/preds/${method}/ep${ep} --ckpt ckpt/${method}/ep${ep}.pth --size ${size}
done


# Eval
cd evaluation
python main.py --methods ${method}
python sort_results.py
cd ..



nvidia-smi
hostname
