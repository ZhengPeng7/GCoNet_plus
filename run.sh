# Run python script
method="gconet_1"
size=224

# Train
python train.py --loss DSLoss_IoU_noCAM --trainset DUTS_class --size ${size} --ckpt_dir ckpt/${method} --lr 3e-4 --bs 48 --epochs 55

# # Test, --ckpt ckpt/gconet/epxx.pth, final == epochs - 1, don't do it twice
# python test.py --pred_dir /home/pz1/datasets/sod/preds/${method}/final --ckpt ckpt/${method}/final.pth

for ep in {45,46,47,48,49,50,51,52,53,54}
do
python test.py --pred_dir /home/pz1/datasets/sod/preds/${method}/ep${ep} --ckpt ckpt/${method}/ep${ep}.pth --size ${size}
done


# Eval
cd evaluation
python main.py --methods ${method}
python sort_results.py
cd ..
