id_start=49
id_len=3
for ((pred_id=${id_start};pred_id<${id_start}+${id_len};pred_id++))
do
sbatch gco.sh ${pred_id}
done
