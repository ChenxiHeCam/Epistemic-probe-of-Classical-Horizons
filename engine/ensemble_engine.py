"""ENSEMBLE ENGINE: the detector is not a new statistic — it is a notation-free physics framework whose
engine COMBINES established off-the-shelf anomaly detectors. Here: breakdown-ratio + split-conformal,
aggregated by rank-average (unsupervised). Show the ensemble is at least as good as, and more robust than,
either alone, on the fair larger synthetic suite and on the real-data triad."""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
clp=lambda z:np.clip(z,0,60); C=3.0
FORMS={'lin':(lambda x,a,b:a*x+b,[1,0]),'pow':(lambda x,a,b,c:a*np.abs(x)**b+c,[1,1,0]),
 'exp':(lambda x,a,b,c:a*np.exp(np.clip(-b*x,-60,60))+c,[1,.5,0]),'quad':(lambda x,a,b,c:a*x**2+b*x+c,[1,0,0]),
 'cubic':(lambda x,a,b,c,d:a*x**3+b*x**2+c*x+d,[1,0,0,0]),'sqrt':(lambda x,a,b:a*np.sqrt(np.abs(x))+b,[1,0]),'const':(lambda x,a:a+0*x,[1])}
def fit_auto(x,y,tr):
    best=1e9;bp=None
    for fm,(f,p0) in FORMS.items():
        try:
            p,_=curve_fit(f,x[tr],y[tr],p0=p0,maxfev=6000);r=np.sqrt(np.mean((f(x[tr],*p)-y[tr])**2))
            if np.isfinite(r) and r<best: best=r;bp=(f,p)
        except: pass
    return bp
def sp(x,frac=0.35):
    o=np.argsort(x);n=len(x);return o,o[:int(frac*n)],o[int(0.7*n):]
def breakdown(x,y):
    o,tr,te=sp(x);bp=fit_auto(x,y,tr)
    if bp is None: return np.nan
    f,p=bp;pred=f(x,*p);pos=(y>0)&(np.abs(pred)>1e-9)
    if pos.sum()<8: return np.nan
    lr=np.abs(np.log(np.abs(y[pos])/np.abs(pred[pos])));L=np.full(len(x),np.nan);L[pos]=lr
    return np.nanmedian(L[te])/(np.nanmedian(L[tr])+1e-6)
def conformal(x,y):
    o,tr,te=sp(x);bp=fit_auto(x,y,tr)
    if bp is None: return np.nan
    f,p=bp;res=np.abs(y-f(x,*p));q=np.quantile(res[tr],0.9)+1e-9
    return float(np.mean(res[te]/q))
def mk(seed):
    r=np.random.RandomState(seed);phen=[]
    cl=[lambda x:2*x,lambda x:.5*x**2,lambda x:x**1.5,lambda x:1/x**2,lambda x:x**4,lambda x:np.exp(-x/2)+.02,
        lambda x:3*np.exp(-.5*x)+.05,lambda x:2*np.sqrt(x),lambda x:5/x,lambda x:3*x,lambda x:x**2,lambda x:9.8/x**2,
        lambda x:x**2.5,lambda x:np.log(x+1),lambda x:1-np.exp(-x),lambda x:x,lambda x:x**3,lambda x:1/(x+.5)]
    for fn in cl:
        x=np.sort(r.uniform(0.2,8,200));y=fn(x)*(1+0.02*r.randn(200));phen.append((x,y,0))
    nw=[lambda x:x**3/(np.exp(clp(x/1.5))-1+1e-9),lambda x:x/(np.exp(clp(x/1.5))-1+1e-9),
        lambda x:(1/np.sqrt(np.clip(1-(x/C)**2,1e-4,1))-1),lambda x:x/np.sqrt(np.clip(1-(x/C)**2,1e-4,1)),
        lambda x:C**2/np.sqrt(np.clip(1-(x/C)**2,1e-4,1)),lambda x:1/(np.exp(np.clip((x-1.5)/.15,-60,60))+1),
        lambda x:(lambda z:z**2*np.exp(z)/(np.exp(z)-1)**2)(np.clip(2./np.clip(x,1e-3,None),1e-3,40)),
        lambda x:np.maximum(x-1.2,0)+1e-3,lambda x:1/np.sqrt(np.clip(1-(x/3.2)**2,1e-4,1)),
        lambda x:x**2/np.sqrt(np.clip(1-(x/3.5)**2,1e-4,1)),lambda x:x**2.5/(np.exp(clp(x/2))-1+1e-9),
        lambda x:1/(np.exp(clp(x/1.2))-1+1e-9),lambda x:1/(np.exp(np.clip((x-2)/.1,-60,60))+1),
        lambda x:(1/np.sqrt(np.clip(1-(x/2.8)**2,1e-4,1))-1),lambda x:x/np.sqrt(np.clip(1-(x/2.6)**2,1e-4,1)),
        lambda x:x**3/(np.exp(clp(x/1.0))-1+1e-9),lambda x:x**4/(np.exp(clp(x/2.5))-1+1e-9),lambda x:np.maximum(x-0.8,0)+1e-3]
    rngs=[(.05,14),(.05,12),(.05,2.98),(.05,2.98),(.05,2.98),(0,3),(.1,8),(0,4),(.05,3.1),(.05,3.4),(.1,9),(.1,10),(0,3),(.05,2.75),(.05,2.55),(.05,10),(.1,11),(0,4)]
    for fn,rng in zip(nw,rngs):
        x=np.sort(r.uniform(*rng,200));y=fn(x)*(1+0.02*r.randn(200));m=np.isfinite(y)&(np.abs(y)<1e10);phen.append((x[m],y[m],1))
    return phen
phen=mk(3)
bd=np.array([breakdown(x,y) for x,y,_ in phen]); cf=np.array([conformal(x,y) for x,y,_ in phen])
lab=np.array([l for _,_,l in phen]); ok=np.isfinite(bd)&np.isfinite(cf)
bd,cf,lab=bd[ok],cf[ok],lab[ok]
# rank-average ensemble (unsupervised: just order-combine the two anomaly scores)
ens=(rankdata(bd)+rankdata(cf))/2.0
def ci(s,y):
    a=roc_auc_score(y,s);r=np.random.RandomState(0);b=[]
    for _ in range(3000):
        ic=r.choice(np.where(y==0)[0],np.sum(y==0),True);inw=r.choice(np.where(y==1)[0],np.sum(y==1),True)
        idx=np.concatenate([ic,inw]);b.append(roc_auc_score(y[idx],s[idx]))
    return a,np.percentile(b,2.5),np.percentile(b,97.5)
print('=== ENSEMBLE ENGINE (breakdown-ratio + conformal, rank-average) — fair suite n=%d ==='%len(lab))
for nm,s in [('breakdown-ratio alone',bd),('conformal alone',cf),('ENSEMBLE (rank-avg)',ens)]:
    a,lo,hi=ci(s,lab); print('  %-24s AUROC %.3f  95%% CI [%.3f, %.3f]'%(nm,a,lo,hi))
# robustness: min AUROC across 5 different random suites (does ensemble reduce variance?)
print('\n  robustness across 5 random suites (min / mean AUROC):')
for nm,fn in [('breakdown',breakdown),('conformal',conformal)]:
    aus=[]
    for sd in range(5):
        ph=mk(sd);s=np.array([fn(x,y) for x,y,_ in ph]);L=np.array([l for _,_,l in ph]);o=np.isfinite(s)
        aus.append(roc_auc_score(L[o],s[o]))
    print('    %-12s min %.3f  mean %.3f'%(nm,min(aus),np.mean(aus)))
ens_aus=[]
for sd in range(5):
    ph=mk(sd);b=np.array([breakdown(x,y) for x,y,_ in ph]);c=np.array([conformal(x,y) for x,y,_ in ph]);L=np.array([l for _,_,l in ph])
    o=np.isfinite(b)&np.isfinite(c);e=(rankdata(b[o])+rankdata(c[o]))/2;ens_aus.append(roc_auc_score(L[o],e))
print('    %-12s min %.3f  mean %.3f'%('ENSEMBLE',min(ens_aus),np.mean(ens_aus)))
