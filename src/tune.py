import sys, json
from common import *
from scipy.optimize import differential_evolution
from math import gamma as G, sin, pi
import os
LB=np.array([-5.,-10.]); UB=np.array([10.,3.]); BUDGET=int(os.environ.get('BUDGET',100))
class Obj:
    def __init__(s,X,y,seed): s.X,s.y,s.seed,s.n,s.best,s.bx=X,y,seed,0,-9,None
    def __call__(s,x):
        x=np.clip(x,LB,UB)
        if s.n>=BUDGET: return 9.0
        s.n+=1; v=inner_score(s.X,s.y,x[0],x[1],s.seed)
        if v>s.best: s.best,s.bx=v,x.copy()
        return -v
def t_grid(o,rng):
    for a in np.linspace(LB[0],UB[0],10):
        for b in np.linspace(LB[1],UB[1],10): o(np.array([a,b]))
def t_random(o,rng):
    for _ in range(BUDGET): o(LB+rng.random(2)*(UB-LB))
def t_de(o,rng):
    # pop = 5*2 = 10, 9 generations + init = 100 evaluations, DE/rand/1/bin
    differential_evolution(o,list(zip(LB,UB)),strategy='rand1bin',popsize=5,maxiter=BUDGET//10-1,mutation=(0.5,1.0),
        recombination=0.7,init='random',polish=False,tol=0,seed=int(rng.integers(1e9)))
def levy(rng,d,beta=1.5):
    sig=(G(1+beta)*sin(pi*beta/2)/(G((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    return 0.01*rng.normal(0,1,d)*sig/np.abs(rng.normal(0,1,d))**(1/beta)
def t_hho(o,rng,N=10):
    # Harris Hawks Optimization (Heidari et al., 2019), stopped at the same evaluation budget
    d=2; P=LB+rng.random((N,d))*(UB-LB); F=np.array([o(p) for p in P])
    T=(BUDGET-N)//N; 
    for t in range(T*3):
        if o.n>=BUDGET: break
        i=np.argmin(F); R,FR=P[i].copy(),F[i]
        E1=2*(1-t/max(T,1)) if t<T else 0.05
        for k in range(N):
            if o.n>=BUDGET: break
            E=E1*(2*rng.random()-1)
            if abs(E)>=1:
                q=rng.random(); r=rng.integers(N)
                if q>=0.5: Xn=P[r]-rng.random()*np.abs(P[r]-2*rng.random()*P[k])
                else: Xn=(R-P.mean(0))-rng.random()*(LB+rng.random()*(UB-LB))
                Xn=np.clip(Xn,LB,UB); f=o(Xn)
                if f<F[k]: P[k],F[k]=Xn,f
            else:
                r=rng.random(); J=2*(1-rng.random())
                if r>=0.5 and abs(E)>=0.5: Xn=(R-P[k])-E*np.abs(J*R-P[k])
                elif r>=0.5: Xn=R-E*np.abs(R-P[k])
                else:
                    base=R-E*np.abs(J*R-P[k]) if abs(E)>=0.5 else R-E*np.abs(J*R-P.mean(0))
                    Y=np.clip(base,LB,UB); fy=o(Y)
                    if fy<F[k]: P[k],F[k]=Y,fy; continue
                    Z=np.clip(Y+rng.random(d)*levy(rng,d),LB,UB); fz=o(Z)
                    if fz<F[k]: P[k],F[k]=Z,fz
                    continue
                Xn=np.clip(Xn,LB,UB); f=o(Xn)
                if f<F[k]: P[k],F[k]=Xn,f
TUNERS={'Grid':t_grid,'Random':t_random,'DE':t_de,'HHO':t_hho}
if os.environ.get('ONLY'): TUNERS={k:TUNERS[k] for k in os.environ['ONLY'].split(',')}
def run(name):
    X,y=load(name); out=[]
    rs=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=42)
    for f,(a,b) in enumerate(rs.split(X,y)):
        Xa,ya,Xb,yb=X[a],y[a],X[b],y[b]
        if os.environ.get('SKIPDEF'): pass
        else:
         # default SVM (C=1, gamma=scale ~ 1/n_features after scaling)
         t=time.perf_counter(); p=svm_pipe(0,np.log2(1/20),seed=f).fit(Xa,ya); el=time.perf_counter()-t
         m=metrics(yb,p.predict(Xb),p.decision_function(Xb)); m.update(ds=name,fold=f,tuner='Default',time=el,evals=0,lc=0,lg=np.log2(1/20)); out.append(m)
        for tn,fn in TUNERS.items():
            rng=np.random.default_rng(1000*f+7); o=Obj(Xa,ya,f); t=time.perf_counter(); fn(o,rng)
            p=svm_pipe(o.bx[0],o.bx[1],seed=f).fit(Xa,ya); el=time.perf_counter()-t
            m=metrics(yb,p.predict(Xb),p.decision_function(Xb)); m.update(ds=name,fold=f,tuner=tn,time=el,evals=o.n,lc=float(o.bx[0]),lg=float(o.bx[1]),inner=o.best); out.append(m)
        print(name,f,flush=True)
    sfx='' if BUDGET==100 and not os.environ.get('ONLY') else f'_b{BUDGET}'
    pd.DataFrame(out).to_csv(os.path.join(RES_DIR, f'res_tune_{name}{sfx}.csv'),index=False)
if __name__=='__main__':
    for n in sys.argv[1:]: run(n)
