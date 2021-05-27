import numpy as np


with open('output/details/result.txt', 'r') as f:
    res = f.read()

res = res.replace('||', '').replace('(', '').replace(')', '')

score = []
for r in res.splitlines():
    ds = r.split()
    s = ds[:2]
    for idx_d, d in enumerate(ds[2:]):
        if idx_d % 2 == 0:
            s.append(float(d))
    score.append(s)

ss = sorted(score, key=lambda x: (x[0], x[2], x[3], x[4], x[5], x[1]), reverse=True)
ss_ar = np.array(ss)
np.savetxt('score_sorted.txt', ss_ar, fmt='%s')
ckpt_coca = ss_ar[ss_ar[:, 0] == 'CoCA'][0][1]
ckpt_cosod = ss_ar[ss_ar[:, 0] == 'CoSOD3k'][0][1]
ckpt_cosal = ss_ar[ss_ar[:, 0] == 'CoSal2015'][0][1]

best_coca_scores = ss_ar[ss_ar[:, 1] == ckpt_coca]
best_cosod_scores = ss_ar[ss_ar[:, 1] == ckpt_cosod]
best_cosal_scores = ss_ar[ss_ar[:, 1] == ckpt_cosal]
print('Best (models may be different):')
print('CoCA:\n', best_coca_scores)
print('CoSOD3k:\n', best_cosod_scores)
print('CoSal2015:\n', best_cosal_scores)
