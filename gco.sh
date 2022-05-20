#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00


# Run python script
method="gconet_$1"
size=256
epochs=350
val_last=50

# Train
CUDA_VISIBLE_DEVICES=0 python train.py --trainset DUTS_class --size ${size} --ckpt_dir ckpt/${method} --epochs ${epochs} --val_dir tmp4val_${method}

# # # Show validation results
# # python collect_bests.py

# Test
for ((ep=${epochs}-${val_last};ep<${epochs};ep++))
do
pred_dir=/root/datasets/sod/preds/${method}/ep${ep}
rm -rf ${pred_dir}
CUDA_VISIBLE_DEVICES=0 python test.py --pred_dir ${pred_dir} --ckpt ckpt/${method}/ep${ep}.pth --size ${size}
done


# Eval
cd evaluation
CUDA_VISIBLE_DEVICES=0 python main.py --methods ${method}
python sort_results.py
python select_results.py
cd ..



nvidia-smi
hostname
