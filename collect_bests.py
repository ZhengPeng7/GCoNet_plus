import os

output_files = sorted([sl for sl in os.listdir('.') if 'slurm' in sl])

search_sentence = 'Best epoch is epoch-'
for idx_output_file, output_file in enumerate(output_files):
    with open(output_file, 'r') as fout:
        output = [ln for ln in fout.readlines() if search_sentence in ln][-1]
    best_epoch = int(output.split(search_sentence)[-1].split()[0])
    best_emax = float(output.split()[-1])
    print('Best E_max of {}-th trial is {} in epoch {}'.format(idx_output_file+1, best_emax, best_epoch))

