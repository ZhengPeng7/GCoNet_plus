method="GCoNet_ext"

# Train
rm ckpt/*

# Eval
rm -r output

# Test -- too slow, put at the last 
rm -r /home/pz1/datasets/sod/preds/${method}
