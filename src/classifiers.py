from common import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
def C(): return {'DT':DecisionTreeClassifier(random_state=0),'LR':LogisticRegression(max_iter=2000),
 'SVM':SVC(),'RF':RandomForestClassifier(n_estimators=200,random_state=0),'KNN':KNeighborsClassifier(5),
 'NB':GaussianNB(),'MLP':MLPClassifier((50,),max_iter=1000,random_state=0),'Majority':DummyClassifier(strategy='most_frequent')}
out=[]
for name in ORDER:
    X,y=load(name)
    for f,(a,b) in enumerate(RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=42).split(X,y)):
        for sm in (False,True):
            for cn,clf in C().items():
                if sm and cn=='Majority': continue
                steps=[('sc',StandardScaler())]+([('sm',SMOTE(random_state=f,k_neighbors=3))] if sm else [])+[('clf',clf)]
                p=Pipeline(steps).fit(X[a],y[a]); pr=p.predict(X[b])
                sc=p.decision_function(X[b]) if hasattr(p,'decision_function') and cn in('SVM','LR') else p.predict_proba(X[b])[:,1]
                m=metrics(y[b],pr,sc); m.update(ds=name,fold=f,clf=cn,smote=sm); out.append(m)
    print(name,flush=True)
pd.DataFrame(out).to_csv(os.path.join(RES_DIR, 'res_clf.csv'),index=False)
