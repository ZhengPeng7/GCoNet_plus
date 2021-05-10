method="GCoNet_ext"

# Train
python train.py --loss DSLoss_IoU_noCAM --trainset DUTS_class --size 224 --ckpt_dir ckpt --lr 1e-4 --bs 16 --epochs 5

# Test, --ckpt ckpt/gconet_epxx.pth, final == epochs - 1, don't do it twice
python test.py --pred_dir /home/pz1/datasets/sod/preds/${method}/final --ckpt ckpt/gconet_final.pth

# ep -> 0 ~ N-1
for ep in {3,4}
do
python test.py --pred_dir /home/pz1/datasets/sod/preds/${method}/ep${ep} --ckpt ckpt/gconet_ep${ep}.pth
done


# Eval
cd evaluation
python main.py --methods ${method}
cd ..
