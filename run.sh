method="gconet"

# Train
python train.py --loss DSLoss_IoU_noCAM --trainset DUTS_class --size 224 --ckpt_dir ckpt/${method} --lr 1e-4 --bs 16 --epochs 55

# # Test, --ckpt ckpt/gconet/epxx.pth, final == epochs - 1, don't do it twice
python test.py --pred_dir /home/pz1/datasets/sod/preds/${method}/final --ckpt ckpt/${method}/final.pth

for ep in {46,47,48,49,50,51,52,53}
do
python test.py --pred_dir /home/pz1/datasets/sod/preds/${method}/ep${ep} --ckpt ckpt/${method}/ep${ep}.pth
done


# Eval
cd evaluation
python main.py --methods ${method}
cd ..
