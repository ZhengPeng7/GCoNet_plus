method="GCoNet_ext"

# Train
python train.py--loss DSLoss_IoU_noCAM --trainset DUTS_class --size 224 --ckpt ckpt --lr 1e-4 --bs 16 --epochs 50

# Test
python test.py --param_root ckpt --pred_dir /home/pz1/datasets/sod/${method}/preds

# Eval
cd evaluation
python main.py --gt_dir /home/pz1/datasets/sod/gts --pred_dir /home/pz1/datasets/sod/${method}/preds
cd ..
