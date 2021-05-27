method="gconet"

# Train
rm slurm*
rm -r ckpt

# Eval
rm -r evaluation/output
rm -r evaluation/score_sorted.txt

# # Test -- too slow, put at the last 
# rm -r /home/pz1/datasets/sod/preds/${method}*
