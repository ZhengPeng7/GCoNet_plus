import os
import matplotlib.pyplot as plt
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

ss = sorted(score, key=lambda x: (x[0], x[2], x[3], x[4], x[1]), reverse=True)
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

model_indices = sorted([fname.split('_')[-1] for fname in os.listdir('output/details') if 'gconet_' in fname])
emax = {}
for model_idx in model_indices:
    m = 'gconet_{}-'.format(model_idx)
    if m not in list(emax.keys()):
        emax[m] = []
    for s in score:
        if m in s[1]:
            ep = int(s[1].split('ep')[-1].rstrip('):'))
            emax[m].append([ep, s[2], s[0]])
for m, e in emax.items():
    plot_name = m[:-1]
    print('Saving {} ...'.format(plot_name))
    e = np.array(e)
    e_coca = e[e[:, -1] == 'CoCA']
    e_cosod = e[e[:, -1] == 'CoSOD3k']
    e_cosal = e[e[:, -1] == 'CoSal2015']
    eps = sorted(list(set(e_coca[:, 0].astype(float))))

    e_coca = np.array(sorted(e_coca, key=lambda x: int(x[0])))[:, 1].astype(float)
    e_cosod = np.array(sorted(e_cosod, key=lambda x: int(x[0])))[:, 1].astype(float)
    e_cosal = np.array(sorted(e_cosal, key=lambda x: int(x[0])))[:, 1].astype(float)

    plt.figure()
    plt.plot(eps, e_coca)
    plt.plot(eps, e_cosod)
    plt.plot(eps, e_cosal)
    plt.legend(['CoCA', 'CoSOD3k', 'CoSal2015'])
    plt.title(m)
    plt.savefig('{}.png'.format(plot_name))
