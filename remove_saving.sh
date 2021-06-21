method="gconet"

# Val
rm -r tmp4val*

# Train
rm slurm*
rm -r ckpt

# Eval
rm -r evaluation/output
rm -r evaluation/score_sorted.txt
rm -r evaluation/${method}_*.png

# # Test -- too slow, put at the last 
# rm -r /home/pz1/datasets/sod/preds/${method}*
