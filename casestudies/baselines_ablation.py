"""(1) BASELINES — compare our breakdown-ratio against standard model-misspecification
tests (Ramsey RESET F-test; CUSUM change-point on residuals) on the same benchmark. (2) ORACLE ABLATION —
our method when the classical law is GIVEN vs auto-selected from a dictionary vs auto-selected WITH the
interior/valid regime also unknown (auto-split). Report AUROC (needs-new vs classical) for each."""
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
clp=lambda z:np.clip(z,0,60); C=3.0
# 1-D phenomena: (name, fn, x-range, known-classical-form-name, label 1=needs-new)
PH=[
 ('Ohm',            lambda x:2*x,                    (.1,8),'lin',0),
 ('KE v^2',         lambda x:0.5*x**2,               (.1,8),'pow',0),
 ('Kepler a^1.5',   lambda x:x**1.5,                 (.2,8),'pow',0),
 ('Coulomb 1/r^2',  lambda x:1/x**2,                 (.3,6),'pow',0),
 ('Stefan T^4',     lambda x:x**4,                   (.3,5),'pow',0),
 ('RC discharge',   lambda x:np.exp(-x/2)+0.02,      (0,8),'exp',0),
 ('Newton cooling', lambda x:3*np.exp(-.5*x)+.05,    (0,8),'exp',0),
 ('pendulum sqrt',  lambda x:2*np.sqrt(x),           (.1,8),'pow',0),
 ('blackbody',      lambda x:x**3/(np.exp(clp(x/1.5))-1+1e-9),(.05,14),'pow',1),
 ('Planck osc-E',   lambda x:x/(np.exp(clp(x/1.5))-1+1e-9),(.05,12),'pow',1),
 ('rel KE',         lambda x:(1/np.sqrt(np.clip(1-(x/C)**2,1e-4,1))-1),(.05,2.985),'pow',1),
 ('rel momentum',   lambda x:x/np.sqrt(np.clip(1-(x/C)**2,1e-4,1)),(.05,2.985),'lin',1),
 ('rel energy',     lambda x:C**2/np.sqrt(np.clip(1-(x/C)**2,1e-4,1)),(.05,2.985),'const',1),
 ('tunneling',      lambda x:np.exp(-2*clp(x)),      (.02,4),'exp',1),
 ('Fermi step',     lambda x:1/(np.exp(np.clip((x-1.5)/.15,-60,60))+1),(0,3),'lin',1),
 ('specific heat',  lambda x:(lambda z:z**2*np.exp(z)/(np.exp(z)-1)**2)(np.clip(2.0/np.clip(x,1e-3,None),1e-3,40)),(.1,8),'const',1),
]
FORMS={'lin':(lambda x,a,b:a*x+b,[1,0]),'pow':(lambda x,a,b,c:a*np.abs(x)**b+c,[1,1,0]),
 'exp':(lambda x,a,b,c:a*np.exp(np.clip(-b*x,-60,60))+c,[1,.5,0]),'const':(lambda x,a:a+0*x,[1]),
 'quad':(lambda x,a,b,c:a*x**2+b*x+c,[1,0,0]),'cubic':(lambda x,a,b,c,d:a*x**3+b*x**2+c*x+d,[1,0,0,0]),
 'sqrt':(lambda x,a,b:a*np.sqrt(np.abs(x))+b,[1,0])}
def fit(form,x,y):
    f,p0=FORMS[form]
    try: p,_=curve_fit(f,x,y,p0=p0,maxfev=6000); return f,p
    except: return f,p0
def split(x,fe='low',frac=0.35):
    o=np.argsort(x);n=len(x);tr=o[:int(frac*n)] if fe=='low' else o[int((1-frac)*n):]; te=o[int(.7*n):] if fe=='low' else o[:int(.3*n)]
    return o,tr,te
# --- our breakdown ratio ---
def breakdown(x,y,form,fe='low'):
    o,tr,te=split(x,fe); f,p=fit(form,x[tr],y[tr]); pred=f(x,*p); pos=(y>0)&(np.abs(pred)>1e-9)
    lr=np.abs(np.log(np.abs(y[pos])/np.abs(pred[pos]))); L=np.full(len(x),np.nan); L[pos]=lr
    ir=np.nanmedian(L[tr])+1e-6; fr=np.nanmedian(L[te]); return fr/ir
def breakdown_auto(x,y,fe='low',known=None):
    forms=[known] if known else ['lin','pow','exp','quad','cubic','sqrt','const']
    o,tr,te=split(x,fe); best=1e9;bp=None
    for fm in forms:
        f,p=fit(fm,x[tr],y[tr]); r=np.sqrt(np.mean((f(x[tr],*p)-y[tr])**2))
        if np.isfinite(r) and r<best: best=r;bp=(f,p)
    f,p=bp; pred=f(x,*p); pos=(y>0)&(np.abs(pred)>1e-9)
    lr=np.abs(np.log(np.abs(y[pos])/np.abs(pred[pos]))); L=np.full(len(x),np.nan); L[pos]=lr
    ir=np.nanmedian(L[tr])+1e-6; fr=np.nanmedian(L[te]); return fr/ir
def breakdown_autosplit(x,y):  # neither law NOR regime given: try both ends, take max breakdown
    return max(breakdown_auto(x,y,'low'),breakdown_auto(x,y,'high'))
# --- baseline 1: Ramsey RESET (add powers of fitted value, F-test) ---
def reset(x,y,form,fe='low'):
    o,tr,te=split(x,fe); f,p=fit(form,x[tr],y[tr]); yhat=f(x,*p)
    # restricted: y ~ yhat ; unrestricted: y ~ yhat + yhat^2 + yhat^3
    X0=np.vstack([yhat,np.ones_like(yhat)]).T; b0,_,_,_=np.linalg.lstsq(X0,y,rcond=None); r0=y-X0@b0
    X1=np.vstack([yhat,yhat**2,yhat**3,np.ones_like(yhat)]).T; b1,_,_,_=np.linalg.lstsq(X1,y,rcond=None); r1=y-X1@b1
    ss0=np.sum(r0**2);ss1=np.sum(r1**2);q=2;n=len(y);k=4
    F=((ss0-ss1)/q)/((ss1)/(n-k)+1e-12); return F  # large F = misspecification
# --- baseline 2: CUSUM change-point on classical-fit residuals ---
def cusum(x,y,form,fe='low'):
    o,tr,te=split(x,fe); f,p=fit(form,x[tr],y[tr]); res=(y-f(x,*p)); res=res[np.argsort(x)]
    s=np.cumsum((res-np.mean(res))/(np.std(res)+1e-9)); return np.max(np.abs(s))/np.sqrt(len(res))
# run
def score_all(scorer,**kw):
    ys=[];ss=[]
    for name,fn,rng,form,lab in PH:
        r=np.random.RandomState(3);x=np.sort(r.uniform(*rng,220));y=fn(x)
        m=np.isfinite(y)&(np.abs(y)<1e10);x=x[m];y=y[m]
        fe='high' if name in ('rel energy','specific heat') else 'low'
        try: s=scorer(x,y,form,fe) if 'form' in scorer.__code__.co_varnames else scorer(x,y)
        except Exception: s=np.nan
        if np.isfinite(s): ys.append(lab);ss.append(s)
    return roc_auc_score(ys,ss)
print('=== BASELINES vs OUR METHOD, and ORACLE ABLATION (AUROC needs-new vs classical, %d phenomena) ===\n'%len(PH))
print('  %-46s AUROC'%'method')
print('  %-46s %.3f'%('OUR breakdown-ratio (classical law GIVEN)',score_all(lambda x,y,form,fe:breakdown(x,y,form,fe))))
print('  %-46s %.3f'%('OUR breakdown-ratio (law AUTO-selected)',score_all(lambda x,y:breakdown_auto(x,y,'low'))))
print('  %-46s %.3f'%('OUR breakdown (law + regime BOTH unknown)',score_all(lambda x,y:breakdown_autosplit(x,y))))
print('  %-46s %.3f'%('BASELINE Ramsey RESET F-test',score_all(lambda x,y,form,fe:reset(x,y,form,fe))))
print('  %-46s %.3f'%('BASELINE CUSUM change-point',score_all(lambda x,y,form,fe:cusum(x,y,form,fe))))
print('\n  -> shows (a) how our statistic compares to standard misspecification tests,')
print('     (b) how much performance degrades as oracle knowledge (law, then regime) is removed.')
