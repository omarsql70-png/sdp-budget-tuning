import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, glob, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES_DIR, ORDER
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
from matplotlib.patches import FancyBboxPatch
plt.rcParams['font.family']='DejaVu Serif'
# Fig 1 vertical, column width
fig,ax=plt.subplots(figsize=(3.4,3.6),dpi=300); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,112)
def box(x,y,w,h,t,fc,fs=6.0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=2',fc=fc,ec='#333',lw=0.7))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def arr(x1,y1,x2,y2): ax.annotate('',(x2,y2),(x1,y1),arrowprops=dict(arrowstyle='->',lw=0.8,color='#333'))
box(6,100,88,10,'8 PROMISE releases (20 CK/OO metrics,\ndefective = bug count > 0)','#e8eef7')
box(6,84,88,10,'Outer loop: stratified 5-fold CV × 2 repeats\n→ 10 held-out test folds per release','#e8eef7')
box(2,58,44,20,'Part A\n7 classifiers + majority\nbaseline, with and\nwithout SMOTE','#f3f3f3')
box(54,58,44,20,'Part B: RBF-SVM tuning\nDefault | Grid | Random\n| DE | HHO\n(100 evaluations each)','#fbeee0')
box(52,32,46,18,'Inner 3-fold CV on the\ntraining fold only\n(z-score → SMOTE → SVM)\nfitness = mean MCC','#fbeee0')
box(54,14,44,10,'Refit best (C, γ) on\nthe full training fold','#f3f3f3')
box(2,0,96,8,'Held-out fold: MCC, AUC, F1, precision, recall, time','#e6f2e6')
arr(50,100,50,94.6); arr(35,84,24,78.6); arr(65,84,76,78.6); arr(76,58,76,50.6); arr(76,32,76,24.6)
arr(24,58,24,8.6); arr(76,14,76,8.6)
plt.savefig(os.path.join(FIG_DIR,'fig1_protocol.png'),bbox_inches='tight',dpi=300); plt.close()
# Fig 2 inner vs outer
d=pd.concat(pd.read_csv(f) for f in [os.path.join(RES_DIR, f'res_tune_{n}.csv') for n in ORDER]); t=d[d.tuner!='Default']
order=['Grid','Random','DE','HHO']; g=t.groupby('tuner')[['inner','mcc']].mean().loc[order]
fig,ax=plt.subplots(figsize=(3.4,2.0),dpi=300); x=np.arange(4); w=0.36
b1=ax.bar(x-w/2,g['inner'],w,label='Inner-CV MCC (used for selection)',color='#9bb7d4',edgecolor='#333',lw=0.5)
b2=ax.bar(x+w/2,g['mcc'],w,label='Held-out MCC',color='#e6a96b',edgecolor='#333',lw=0.5)
dm=d[d.tuner=='Default'].mcc.mean(); ax.axhline(dm,ls='--',lw=0.8,color='k',label=f'Default SVM held-out MCC ({dm:.3f})',zorder=0)
for b in list(b1)+list(b2): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.003,f'{b.get_height():.3f}',ha='center',fontsize=5.6)
ax.set_xticks(x,order,fontsize=7); ax.set_ylim(0.2,0.45); ax.set_ylabel('Mean MCC',fontsize=7); ax.tick_params(labelsize=6.5)
ax.legend(fontsize=6,loc='upper left',frameon=False,ncol=1); ax.spines[['top','right']].set_visible(False)
plt.savefig(os.path.join(FIG_DIR,'fig2_inner_vs_heldout.png'),bbox_inches='tight',dpi=300)
