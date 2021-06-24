id_start=16
id_len=5
for ((pred_id=${id_start};pred_id<${id_start}+${id_len};pred_id++))
do
sbatch gco_mask.sh ${pred_id}
done
