"""Reviewer-concern fixes (local): (1) CONFORMAL-prediction baseline (modern comparator);
(2) LOCALIZATION-ERROR metric on synthetic breakdowns with a KNOWN onset x0; (3) larger synthetic suite
with tighter bootstrap CI. Addresses: 'need modern baselines', 'localization asserted not quantified',
'wide CIs / small suite'."""
import numpy as np
from scipy.optimize import curve_fit
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
def split(x,frac=0.35):
    o=np.argsort(x);n=len(x);return o,o[:int(frac*n)],o[int(0.7*n):]
def breakdown(x,y):
    o,tr,te=split(x);bp=fit_auto(x,y,tr)
    if bp is None: return np.nan
    f,p=bp;pred=f(x,*p);pos=(y>0)&(np.abs(pred)>1e-9)
    if pos.sum()<8: return np.nan
    lr=np.abs(np.log(np.abs(y[pos])/np.abs(pred[pos])));L=np.full(len(x),np.nan);L[pos]=lr
    return np.nanmedian(L[te])/(np.nanmedian(L[tr])+1e-6)
def conformal(x,y):   # split-conformal: interior residual quantile -> frontier nonconformity
    o,tr,te=split(x);bp=fit_auto(x,y,tr)
    if bp is None: return np.nan
    f,p=bp;res=np.abs(y-f(x,*p));q=np.quantile(res[tr],0.9)+1e-9
    return float(np.mean(res[te]/q))   # >1 => frontier outside interior 90% conformal band
# ---- (3) larger synthetic suite ----
def mk(seed):
    r=np.random.RandomState(seed);phen=[]
    # classical (18)
    cl=[lambda x:2*x,lambda x:.5*x**2,lambda x:x**1.5,lambda x:1/x**2,lambda x:x**4,lambda x:np.exp(-x/2)+.02,
        lambda x:3*np.exp(-.5*x)+.05,lambda x:2*np.sqrt(x),lambda x:5/x,lambda x:3*x,lambda x:x**2,lambda x:9.8/x**2,
        lambda x:x**2.5,lambda x:np.log(x+1),lambda x:1-np.exp(-x),lambda x:x,lambda x:x**3,lambda x:1/(x+.5)]
    for i,fn in enumerate(cl):
        rng=(0.2,8);x=np.sort(r.uniform(*rng,200));y=fn(x)*(1+0.02*r.randn(200));phen.append((x,y,0))
    # needs-new (18) with divergence/turnover/threshold/freeze
    nw=[lambda x:x**3/(np.exp(clp(x/1.5))-1+1e-9),lambda x:x/(np.exp(clp(x/1.5))-1+1e-9),
        lambda x:(1/np.sqrt(np.clip(1-(x/C)**2,1e-4,1))-1),lambda x:x/np.sqrt(np.clip(1-(x/C)**2,1e-4,1)),
        lambda x:C**2/np.sqrt(np.clip(1-(x/C)**2,1e-4,1)),lambda x:1/(np.exp(np.clip((x-1.5)/.15,-60,60))+1),
        lambda x:(lambda z:z**2*np.exp(z)/(np.exp(z)-1)**2)(np.clip(2./np.clip(x,1e-3,None),1e-3,40)),
        lambda x:np.maximum(x-1.2,0)+1e-3,lambda x:1/np.sqrt(np.clip(1-(x/3.2)**2,1e-4,1)),
        lambda x:x**2/np.sqrt(np.clip(1-(x/3.5)**2,1e-4,1)),lambda x:x**2.5/(np.exp(clp(x/2))-1+1e-9),
        lambda x:1/(np.exp(clp(x/1.2))-1+1e-9),lambda x:np.tanh(5*(x-2))*0+1/(np.exp(np.clip((x-2)/.1,-60,60))+1),
        lambda x:(1/np.sqrt(np.clip(1-(x/2.8)**2,1e-4,1))-1),lambda x:x/np.sqrt(np.clip(1-(x/2.6)**2,1e-4,1)),
        lambda x:x**3/(np.exp(clp(x/1.0))-1+1e-9),lambda x:x**4/(np.exp(clp(x/2.5))-1+1e-9),lambda x:np.maximum(x-0.8,0)+1e-3]
    rngs=[(.05,14),(.05,12),(.05,2.98),(.05,2.98),(.05,2.98),(0,3),(.1,8),(0,4),(.05,3.1),(.05,3.4),(.1,9),(.1,10),(0,3),(.05,2.75),(.05,2.55),(.05,10),(.1,11),(0,4)]
    for fn,rng in zip(nw,rngs):
        x=np.sort(r.uniform(*rng,200));y=fn(x)*(1+0.02*r.randn(200));m=np.isfinite(y)&(np.abs(y)<1e10);phen.append((x[m],y[m],1))
    return phen
def auroc_of(scorer,phen):
    ys=[];ss=[]
    for x,y,lab in phen:
        s=scorer(x,y)
        if np.isfinite(s): ys.append(lab);ss.append(s)
    return roc_auc_score(ys,ss),ys,ss
phen=mk(3)
a_bd,ys,ss=auroc_of(breakdown,phen); a_cf,_,_=auroc_of(conformal,phen)
# bootstrap CI over phenomena
ss=np.array(ss);ys=np.array(ys);rng=np.random.RandomState(0);boot=[]
for _ in range(3000):
    ic=rng.choice(np.where(ys==0)[0],np.sum(ys==0),replace=True);inw=rng.choice(np.where(ys==1)[0],np.sum(ys==1),replace=True)
    idx=np.concatenate([ic,inw]);boot.append(roc_auc_score(ys[idx],ss[idx]))
print('=== (1)(3) larger synthetic suite (n=%d) + modern baseline ==='%len(phen))
print('  OUR breakdown ratio  AUROC = %.3f   bootstrap 95%% CI [%.3f, %.3f]'%(a_bd,np.percentile(boot,2.5),np.percentile(boot,97.5)))
print('  BASELINE conformal   AUROC = %.3f'%a_cf)
print('  (earlier: GP model-discrepancy 0.58, RESET 0.73, CUSUM 0.67)')
# ---- (2) localization error on synthetic breakdowns with KNOWN onset x0 ----
def localize(x,y):
    o,tr,te=split(x);bp=fit_auto(x,y,tr)
    if bp is None: return np.nan
    f,p=bp;pred=f(x,*p);xs=x[o];ys_=y[o];prd=pred[o]
    lr=np.abs(np.log(np.clip(np.abs(ys_),1e-9,None)/np.clip(np.abs(prd),1e-9,None)))
    for i in range(len(xs)):
        if np.isfinite(lr[i]) and lr[i]>np.log(2): return xs[i]
    return xs[-1]
print('\n=== (2) LOCALIZATION-ERROR: synthetic breakdowns with known onset x0 ===')
errs=[]
for name,fn,rng,x0 in [
    ('classical->diverge @2.0', lambda x:np.where(x<2.0,x**2,x**2*(1+3*(x-2.0)**2)),(0.1,4),2.0),
    ('classical->turnover @3.0',lambda x:np.where(x<3.0,x**2,x**2*np.exp(-(x-3.0))),(0.2,6),3.0),
    ('classical->threshold @1.5',lambda x:np.where(x<1.5,0.0,(x-1.5)),(0,4),1.5),
    ('classical->freeze @1.0',  lambda x:np.where(x>1.0,1.0,np.exp(-(1.0-x)*4)),(0.1,3),1.0),
    ('classical->diverge @2.5', lambda x:np.where(x<2.5,x,x*(1+5*(x-2.5)**2)),(0.1,4),2.5)]:
    r=np.random.RandomState(1);x=np.sort(r.uniform(*rng,200));y=fn(x)*(1+0.02*r.randn(200))
    b=localize(x,y);err=abs(b-x0)/(rng[1]-rng[0]);errs.append(err)
    print('  %-28s true onset %.2f  detected %.2f  error %.1f%% of range'%(name,x0,b,100*err))
print('  --> median localization error = %.1f%% of the variable range (n=%d synthetic onsets)'%(100*np.median(errs),len(errs)))
