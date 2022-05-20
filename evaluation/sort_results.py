import os
import shutil
import matplotlib.pyplot as plt
import numpy as np


move_best_results_here = False

record = ['dataset', 'ckpt', 'Emax', 'Smeasure', 'Fmax', 'MAE', 'Emean', 'Fmean']
measurement = 'Emax'
score_idx = record.index(measurement)

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

ss = sorted(score, key=lambda x: (x[record.index('dataset')], x[record.index('Emax')], x[record.index('Smeasure')], x[record.index('Fmax')], x[record.index('ckpt')]), reverse=True)
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

# Overal relative Emax improvement on three datasets
if measurement == 'Emax':
    gco_scores = {'CoCA': 0.760, 'CoSOD3k': 0.860, 'CoSal2015': 0.887}
    gco_scores_Smeasure = {'CoCA': 0.673, 'CoSOD3k': 0.802, 'CoSal2015': 0.845}
elif measurement == 'Smeasure':
    gco_scores = {'CoCA': 0.673, 'CoSOD3k': 0.802, 'CoSal2015': 0.845}
elif measurement == 'Fmax':
    gco_scores = {'CoCA': 0.544, 'CoSOD3k': 0.777, 'CoSal2015': 0.847}
elif measurement == 'Emean':
    gco_scores = {'CoCA': 0.1, 'CoSOD3k': 0.1, 'CoSal2015': 0.1}
elif measurement == 'Fmean':
    gco_scores = {'CoCA': 0.1, 'CoSOD3k': 0.1, 'CoSal2015': 0.1}
ckpts = list(set(ss_ar[:, 1].squeeze().tolist()))
improvements_mean = []
improvements_lst = []
improvements_mean_Smeasure = []
improvements_lst_Smeasure = []
for ckpt in ckpts:
    scores = ss_ar[ss_ar[:, 1] == ckpt]
    if scores.shape[0] != len(gco_scores):
        improvements_mean.append(-1)
        improvements_lst.append([-1, -1, 1])
        improvements_mean_Smeasure.append(-1)
        improvements_lst_Smeasure.append([-1, -1, 1])
        continue
    score_coca = float(scores[scores[:, 0] == 'CoCA'][0][score_idx])
    score_cosod = float(scores[scores[:, 0] == 'CoSOD3k'][0][score_idx])
    score_cosal = float(scores[scores[:, 0] == 'CoSal2015'][0][score_idx])
    improvements = [
        (score_coca - gco_scores['CoCA']) / gco_scores['CoCA'],
        (score_cosod - gco_scores['CoSOD3k']) / gco_scores['CoSOD3k'],
        (score_cosal - gco_scores['CoSal2015']) / gco_scores['CoSal2015']
    ]
    improvement_mean = np.mean(improvements)
    improvements_mean.append(improvement_mean)
    improvements_lst.append(improvements)

    # Smeasure
    score_coca = float(scores[scores[:, 0] == 'CoCA'][0][record.index('Smeasure')])
    score_cosod = float(scores[scores[:, 0] == 'CoSOD3k'][0][record.index('Smeasure')])
    score_cosal = float(scores[scores[:, 0] == 'CoSal2015'][0][record.index('Smeasure')])
    improvements_Smeasure = [
        (score_coca - gco_scores_Smeasure['CoCA']) / gco_scores_Smeasure['CoCA'],
        (score_cosod - gco_scores_Smeasure['CoSOD3k']) / gco_scores_Smeasure['CoSOD3k'],
        (score_cosal - gco_scores_Smeasure['CoSal2015']) / gco_scores_Smeasure['CoSal2015']
    ]
    improvement_mean_Smeasure = np.mean(improvements_Smeasure)
    improvements_mean_Smeasure.append(improvement_mean_Smeasure)
    improvements_lst_Smeasure.append(improvements_Smeasure)
best_measurement = 'Emax'
if best_measurement == 'Emax':
    best_improvement_index = np.argsort(improvements_mean).tolist()[-1]
    best_ckpt = ckpts[best_improvement_index]
    best_improvement_mean = improvements_mean[best_improvement_index]
    best_improvements = improvements_lst[best_improvement_index]

    best_improvement_mean_Smeasure = improvements_mean_Smeasure[best_improvement_index]
    best_improvements_Smeasure = improvements_lst_Smeasure[best_improvement_index]
elif best_measurement == 'Smeasure':
    best_improvement_index = np.argsort(improvements_mean_Smeasure).tolist()[-1]
    best_ckpt = ckpts[best_improvement_index]
    best_improvement_mean_Smeasure = improvements_mean_Smeasure[best_improvement_index]
    best_improvements_Smeasure = improvements_lst_Smeasure[best_improvement_index]

    best_improvement_mean = improvements_mean[best_improvement_index]
    best_improvements = improvements_lst[best_improvement_index]

print('The overall best one:')
print(ss_ar[ss_ar[:, 1] == best_ckpt])
print('Got Emax improvements on CoCA-{:.3f}%, CoSOD3k-{:.3f}%, CoSal2015-{:.3f}%, mean_improvement: {:.3f}%.'.format(
    best_improvements[0]*100, best_improvements[1]*100, best_improvements[2]*100, best_improvement_mean*100
))
print('Got Smes improvements on CoCA-{:.3f}%, CoSOD3k-{:.3f}%, CoSal2015-{:.3f}%, mean_improvement: {:.3f}%.'.format(
    best_improvements_Smeasure[0]*100, best_improvements_Smeasure[1]*100, best_improvements_Smeasure[2]*100, best_improvement_mean_Smeasure*100
))
trial = int(best_ckpt.split('_')[-1].split('-')[0])
ep = int(best_ckpt.split('ep')[-1].split(':')[0])
if move_best_results_here:
    trial, ep = 'gconet_{}'.format(trial), 'ep{}'.format(ep)
    dr = os.path.join(trial, ep)
    dst = '-'.join((trial, ep))
    shutil.move(os.path.join('/root/datasets/sod/preds', dr), dst)


# model_indices = sorted([fname.split('_')[-1] for fname in os.listdir('output/details') if 'gconet_' in fname])
# emax = {}
# for model_idx in model_indices:
#     m = 'gconet_{}-'.format(model_idx)
#     if m not in list(emax.keys()):
#         emax[m] = []
#     for s in score:
#         if m in s[1]:
#             ep = int(s[1].split('ep')[-1].rstrip('):'))
#             emax[m].append([ep, s[2], s[0]])

# for m, e in emax.items():
#     plot_name = m[:-1]
#     print('Saving {} ...'.format(plot_name))
#     e = np.array(e)
#     e_coca = e[e[:, -1] == 'CoCA']
#     e_cosod = e[e[:, -1] == 'CoSOD3k']
#     e_cosal = e[e[:, -1] == 'CoSal2015']
#     eps = sorted(list(set(e_coca[:, 0].astype(float))))

#     e_coca = np.array(sorted(e_coca, key=lambda x: int(x[0])))[:, 1].astype(float)
#     e_cosod = np.array(sorted(e_cosod, key=lambda x: int(x[0])))[:, 1].astype(float)
#     e_cosal = np.array(sorted(e_cosal, key=lambda x: int(x[0])))[:, 1].astype(float)

#     plt.figure()
#     plt.plot(eps, e_coca)
#     plt.plot(eps, e_cosod)
#     plt.plot(eps, e_cosal)
#     plt.legend(['CoCA', 'CoSOD3k', 'CoSal2015'])
#     plt.title(m)
#     plt.savefig('{}.png'.format(plot_name))
