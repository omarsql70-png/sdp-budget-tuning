import numpy as np, pandas as pd, glob, os, time, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('SDP_DATA', os.path.join(HERE, '..', 'data'))
RES_DIR = os.environ.get('SDP_RESULTS', os.path.join(HERE, '..', 'results'))
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
FEATS=['wmc','dit','noc','cbo','rfc','lcom','ca','ce','npm','lcom3','loc','dam','moa','mfa','cam','ic','cbm','amc','max_cc','avg_cc']
ORDER=['ivy-2.0','redaktor','ant-1.7','xerces-1.3','prop-6','camel-1.6','xalan-2.4','xerces-1.2']
def load(name):
    d=pd.read_csv(os.path.join(DATA_DIR, f'{name}.csv')); d.columns=[c.strip().lower() for c in d.columns]
    X=d[FEATS].astype(float).values; y=(d['bug']>0).astype(int).values
    return X,y
def svm_pipe(lc,lg,smote=True,seed=0):
    steps=[('sc',StandardScaler())]
    if smote: steps.append(('sm',SMOTE(random_state=seed,k_neighbors=3)))
    steps.append(('clf',SVC(C=2.0**lc,gamma=2.0**lg,kernel='rbf',cache_size=500)))
    return Pipeline(steps)
def inner_score(X,y,lc,lg,seed=0):
    sk=StratifiedKFold(3,shuffle=True,random_state=seed); s=[]
    for a,b in sk.split(X,y):
        p=svm_pipe(lc,lg,seed=seed).fit(X[a],y[a]); s.append(matthews_corrcoef(y[b],p.predict(X[b])))
    return float(np.mean(s))
def metrics(y,pred,score):
    return dict(acc=accuracy_score(y,pred),prec=precision_score(y,pred,zero_division=0),rec=recall_score(y,pred),
                f1=f1_score(y,pred),mcc=matthews_corrcoef(y,pred),auc=roc_auc_score(y,score),
                wprec=precision_score(y,pred,average='weighted',zero_division=0),wrec=recall_score(y,pred,average='weighted'),wf1=f1_score(y,pred,average='weighted'))
