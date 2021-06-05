id_start=1
id_len=10
for ((pred_id=${id_start};pred_id<${id_start}+${id_len};pred_id++))
do
sbatch gco.sh ${pred_id}
done
