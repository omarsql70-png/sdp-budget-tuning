"""Recompute every table and statistic reported in the paper from the per-fold result files.

Usage: python src/analysis.py   -> prints a report and writes results/summary.json
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES_DIR, ORDER, load

TUNERS = ['Default', 'Grid', 'Random', 'DE', 'HHO']
CLFS = ['Majority', 'DT', 'LR', 'SVM', 'RF', 'KNN', 'NB', 'MLP']

def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b).sum() - (a[:, None] < b).sum()) / (len(a) * len(b)))

out = {}
# ---- Table I
out['datasets'] = []
for n in ORDER:
    X, y = load(n); out['datasets'].append([n, int(len(y)), int(y.sum()), round(100 * y.mean(), 1)])
print('TABLE I'); [print('  ', r) for r in out['datasets']]

# ---- Table II (Part A)
c = pd.read_csv(os.path.join(RES_DIR, 'res_clf.csv'))
rows = []
for cl in CLFS:
    a = c[(c.clf == cl) & (~c.smote)][['acc', 'mcc', 'rec']].mean()
    b = c[(c.clf == cl) & (c.smote)][['acc', 'mcc', 'auc', 'rec']].mean() if cl != 'Majority' else None
    rows.append([cl] + [round(x, 3) for x in a] + ([round(x, 3) for x in b] if b is not None else [None] * 4))
out['classifiers'] = rows
s = c[(c.smote) & (c.clf != 'Majority')].groupby(['ds', 'clf']).mcc.mean().unstack()
fr = friedmanchisquare(*[s[k] for k in s])
out['clf_friedman'] = [round(fr.statistic, 2), round(fr.pvalue, 3)]
out['clf_ranks'] = s.rank(axis=1, ascending=False).mean().round(2).to_dict()
print('\nTABLE II'); [print('  ', r) for r in rows]
print('  Friedman (SMOTE, 7 classifiers): chi2=%.2f p=%.3f' % tuple(out['clf_friedman']))
print('  mean ranks:', out['clf_ranks'])
print('  accuracy of NB without SMOTE on prop-6 / redaktor:',
      c[(c.clf == 'NB') & (~c.smote)].groupby('ds').acc.mean()[['prop-6', 'redaktor']].round(3).to_dict())
print('  majority-class accuracy range:', c[c.clf == 'Majority'].groupby('ds').acc.mean().round(3).agg(['min', 'max']).to_dict())

# ---- Table III (Part B, budget = 100)
d = pd.concat(pd.read_csv(os.path.join(RES_DIR, f'res_tune_{n}.csv')) for n in ORDER)
m = d.groupby(['ds', 'tuner']).mcc.mean().unstack()[TUNERS].loc[ORDER]
avg = d.groupby('tuner')[['mcc', 'auc', 'f1', 'prec', 'rec', 'acc', 'time']].mean().loc[TUNERS]
out['tuning_per_release'] = [[n] + [round(m.loc[n, t], 3) for t in TUNERS] for n in ORDER]
out['tuning_avg'] = {k: [round(avg.loc[t, k], 3 if k != 'time' else 2) for t in TUNERS] for k in avg.columns}
out['tuning_rank'] = [round(x, 2) for x in m.rank(axis=1, ascending=False).mean()[TUNERS]]
fr = friedmanchisquare(*[m[t] for t in TUNERS]); out['tuning_friedman'] = [round(fr.statistic, 2), round(fr.pvalue, 3)]
p = d.pivot_table(index=['ds', 'fold'], columns='tuner', values='mcc')
pairs = {}
for x, yv in [('Grid', 'Default'), ('Random', 'Default'), ('DE', 'Default'), ('HHO', 'Default'), ('DE', 'HHO')]:
    pairs[f'{x}-{yv}'] = dict(diff=round(float((p[x] - p[yv]).mean()), 3), p=round(float(wilcoxon(p[x], p[yv]).pvalue), 3),
                              delta=round(cliff(p[x], p[yv]), 3))
out['tuning_pairs'] = pairs
t = d[d.tuner != 'Default']
out['optimism'] = t.groupby('tuner')[['inner', 'mcc']].mean().round(3).to_dict()
out['optimism_gap'] = round(float((t.inner - t.mcc).mean()), 3)
out['hparam_sd'] = t.groupby('tuner')[['lc', 'lg']].std().round(2).to_dict()
out['wins_vs_default'] = m[['Grid', 'Random', 'DE', 'HHO']].gt(m['Default'], axis=0).sum().to_dict()
print('\nTABLE III'); [print('  ', r) for r in out['tuning_per_release']]
print('  averages:', out['tuning_avg']); print('  ranks:', out['tuning_rank'])
print('  Friedman: chi2=%.2f p=%.3f' % tuple(out['tuning_friedman']))
for k, v in pairs.items(): print('  ', k, v)
print('  inner vs held-out:', out['optimism'], 'gap', out['optimism_gap'])
print('  SD of selected log2C / log2gamma:', out['hparam_sd'])
print('  releases where tuner beats default:', out['wins_vs_default'])

# ---- Table IV (budget sensitivity), if available
bud = []
for B in (50, 100, 200):
    files = [os.path.join(RES_DIR, f'res_tune_{n}.csv' if B == 100 else f'res_tune_{n}_b{B}.csv') for n in ORDER]
    if not all(os.path.exists(f) for f in files): continue
    db = pd.concat(pd.read_csv(f) for f in files); db = db[db.tuner.isin(['Random', 'DE', 'HHO'])]
    for tn in ['Random', 'DE', 'HHO']:
        x = db[db.tuner == tn]
        pv = x.set_index(['ds', 'fold']).mcc.sub(p['Default']).dropna()
        bud.append(dict(budget=B, tuner=tn, mcc=round(x.mcc.mean(), 3), auc=round(x.auc.mean(), 3),
                        inner=round(x.inner.mean(), 3), gap=round((x.inner - x.mcc).mean(), 3),
                        time=round(x.time.mean(), 2), diff_vs_default=round(float(pv.mean()), 3),
                        p_vs_default=round(float(wilcoxon(pv).pvalue), 3)))
if bud:
    out['budget'] = bud
    print('\nTABLE IV (budget sensitivity)')
    for r in bud: print('  ', r)
json.dump(out, open(os.path.join(RES_DIR, 'summary.json'), 'w'), indent=1)
print('\nwritten', os.path.join(RES_DIR, 'summary.json'))
