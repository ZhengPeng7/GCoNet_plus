method="gconet"

# Val
rm -r tmp*

# Train
rm slurm*
rm -r ckpt
rm *.pth

# Eval
rm -r evaluation/output
rm -r evaluation/score_sorted.txt
rm -r evaluation/${method}_*.png
rm -r evaluation/${method}_*

# # Test -- too slow, put at the last 
# rm -r /root/datasets/sod/preds/${method}*
