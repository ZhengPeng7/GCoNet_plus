#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00


# Run python script
method="GCoNet_ext"

# Train
python train.py--loss DSLoss_IoU_noCAM --trainset DUTS_class --size 224 --ckpt_dir ckpt --lr 1e-4 --bs 16 --epochs 50

# Test
python test.py --ckpt ckpt/gconet_final.pth --pred_dir /home/pz1/datasets/sod/${method}/preds

# Eval
cd evaluation
python main.py --gt_dir /home/pz1/datasets/sod/gts --pred_dir /home/pz1/datasets/sod/${method}/preds
cd ..


nvidia-smi
hostname
