"""HUMAN-MATCHING METHOD: symbolic-fit extrapolation. Fit the interior with a SINGLE classical symbolic
form (chosen as best interior fit from a dictionary), extrapolate it FAITHFULLY to the frontier, compare
to data. Classical forms extrapolate correctly (exp stays exp -> low residual); new physics breaks the
classical form's continuation (Rayleigh-Jeans nu^2 diverges but data peaks -> high residual). 1-D in the
frontier variable = exactly what a physicist does (fit a law in the familiar regime, extrapolate, see it fail)."""
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
clp=lambda z:np.clip(z,0,60); T0=3.0; C=3.0
# 1-D phenomena: y = f(x) over frontier variable x. label 1=needs-new, 0=classical
PH={
 # classical (interior fit should extrapolate correctly -> LOW residual)
 'linear (Ohm)'     :(lambda x: 2*x,                 (0.1,8),0),
 'KE ~v^2'          :(lambda x: 0.5*x**2,            (0.1,8),0),
 'Kepler ~a^1.5'    :(lambda x: x**1.5,              (0.2,8),0),
 'Coulomb ~1/r^2'   :(lambda x: 1/x**2,              (0.3,6),0),
 'Stefan-Boltz ~T^4':(lambda x: x**4,                (0.3,5),0),
 'RC-discharge'     :(lambda x: np.exp(-x/2),        (0,10),0),
 'Newton-cooling'   :(lambda x: 3*np.exp(-0.5*x),    (0,10),0),
 'barometric'       :(lambda x: np.exp(-x/3),        (0,12),0),
 'pendulum ~sqrt'   :(lambda x: np.sqrt(x),          (0.1,8),0),
 'saturating exp'   :(lambda x: 1-np.exp(-x),        (0,8),0),
 # needs-new: classical continuation breaks at the frontier
 'blackbody'        :(lambda x: x**3/(np.exp(clp(x/T0))-1+1e-9), (0.1,20),1),
 'Planck oscillator':(lambda x: x/(np.exp(clp(x/T0))-1+1e-9),    (0.1,15),1),
 'rel-KE'           :(lambda x: (1/np.sqrt(np.clip(1-(x/C)**2,1e-4,1))-1), (0,2.98),1),
 'rel-momentum'     :(lambda x: x/np.sqrt(np.clip(1-(x/C)**2,1e-4,1)),     (0,2.98),1),
 'time-dilation'    :(lambda x: 1/np.sqrt(np.clip(1-(x/C)**2,1e-4,1)),     (0,2.98),1),
 'rel-energy'       :(lambda x: C**2/np.sqrt(np.clip(1-(x/C)**2,1e-4,1)),  (0,2.98),1),
 'tunneling'        :(lambda x: np.exp(-2*clp(x)),   (0.02,4),1),
 'Fermi-Dirac step' :(lambda x: 1/(np.exp(np.clip((x-1.5)/0.15,-60,60))+1),(0,3),1),
}
# dictionary of classical symbolic forms to fit on the interior
FORMS={
 'lin'  :(lambda x,a,b: a*x+b,               [1,0]),
 'quad' :(lambda x,a,b,c: a*x**2+b*x+c,       [1,0,0]),
 'cubic':(lambda x,a,b,c,d: a*x**3+b*x**2+c*x+d,[1,0,0,0]),
 'power':(lambda x,a,b,c: a*np.abs(x)**b+c,   [1,1,0]),
 'exp'  :(lambda x,a,b,c: a*np.exp(np.clip(b*x,-60,60))+c,[1,-0.5,0]),
 'rec'  :(lambda x,a,b,c: a/(x+b)+c,          [1,1,0]),
 'sqrt' :(lambda x,a,b: a*np.sqrt(np.abs(x))+b,[1,0]),
 'log'  :(lambda x,a,b: a*np.log(np.abs(x)+1e-6)+b,[1,0]),
}
def fit_extrap_resid(x,y):
    o=np.argsort(x);x=x[o];y=y[o];n=len(x)
    tr=np.arange(0,int(.6*n));te=np.arange(int(.75*n),n)
    if len(te)<8 or len(tr)<15: return np.nan
    ys=np.std(y)+1e-9
    best_intR=1e9; best_te=np.nan
    for nm,(f,p0) in FORMS.items():
        try:
            popt,_=curve_fit(f,x[tr],y[tr],p0=p0,maxfev=4000)
            fit_tr=f(x[tr],*popt); intR=np.sqrt(np.mean((fit_tr-y[tr])**2))
            if not np.isfinite(intR): continue
            if intR<best_intR:   # choose the classical form that best explains the interior
                pred=f(x[te],*popt); best_te=np.sqrt(np.mean((pred-y[te])**2))/ys; best_intR=intR
        except Exception: continue
    return best_te
import os
NOISE=float(os.environ.get('NOISE','0.0'))   # relative measurement noise
def score(fn,rng,ni=25):
    r=np.random.RandomState(7);res=[]
    for k in range(ni):
        x=np.sort(r.uniform(*rng,200));y=fn(x);m=np.isfinite(y)&(np.abs(y)<1e10)
        x=x[m];y=y[m]
        if len(y)<60: continue
        if NOISE>0: y=y+NOISE*(np.std(y)+1e-9)*r.randn(len(y))   # additive measurement noise
        v=fit_extrap_resid(x,y)
        if np.isfinite(v): res.append(min(v,5))
    return float(np.median(res)) if res else np.nan
rows=[]
for name,(fn,rng,lab) in PH.items(): rows.append((name,lab,score(fn,rng)))
print('=== SYMBOLIC-FIT EXTRAPOLATION detector (fit classical form on interior -> extrapolate -> frontier residual) ===')
print('  novelty = normalized frontier residual of best classical-form fit (HIGH = classical breaks = needs new)\n')
print('  -- CLASSICAL (want LOW) --')
for n,l,s in rows:
    if l==0: print('    %-20s %.3f'%(n,s))
print('  -- NEEDS-NEW (want HIGH) --')
for n,l,s in rows:
    if l==1: print('    %-20s %.3f'%(n,s))
y=[l for _,l,_ in rows];sc=[s for _,_,s in rows]
cl=[s for _,l,s in rows if l==0];nw=[s for _,l,s in rows if l==1]
print('\n  OVERALL AUROC = %.3f'%roc_auc_score(y,sc))
print('  classical median %.3f  vs  needs-new median %.3f'%(np.median(cl),np.median(nw)))
# how many needs-new exceed the max classical (clean-detection threshold)
thr=np.percentile(cl,90); det=sum(1 for s in nw if s>thr)
print('  needs-new detected above 90th-pctile classical (%.3f): %d / %d'%(thr,det,len(nw)))
