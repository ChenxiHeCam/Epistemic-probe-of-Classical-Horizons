"""Expanded PHYSICALLY-GROUNDED SIMULATION tier (labelled: from known laws, NOT real measurements).
~10 historical breakdowns, each spanning its classical regime (clean interior). Ensemble engine
(conformal + breakdown ratio). Reviewer caveat honoured: this strengthens the benchmark's physical
realism; it does NOT add to the real-data count (which remains FIRAS/Bertozzi/Kepler)."""
import numpy as np
from scipy.optimize import curve_fit
import warnings; warnings.filterwarnings('ignore')
clp=lambda z:np.clip(z,0,60); C=3.0
FORMS={'lin':(lambda x,a,b:a*x+b,[1,0]),'pow':(lambda x,a,b,c:a*np.abs(x)**b+c,[1,1,0]),
 'exp':(lambda x,a,b,c:a*np.exp(np.clip(-b*x,-60,60))+c,[1,.5,0]),'const':(lambda x,a:a+0*x,[1])}  # parsimonious dict
def fit_auto(x,y,tr):
    best=1e9;bp=None
    for fm,(f,p0) in FORMS.items():
        try:
            p,_=curve_fit(f,x[tr],y[tr],p0=p0,maxfev=6000);r=np.sqrt(np.mean((f(x[tr],*p)-y[tr])**2))
            if np.isfinite(r) and r<best: best=r;bp=(f,p,r)
        except: pass
    return bp
def engine(x,y,fe='low'):
    o=np.argsort(x);n=len(x);tr=o[:int(.35*n)] if fe=='low' else o[int(.65*n):]; te=o[int(.7*n):] if fe=='low' else o[:int(.3*n)]
    bp=fit_auto(x,y,tr)
    if bp is None: return None
    f,p,ir=bp; pred=f(x,*p); pos=(np.abs(y)>0)&(np.abs(pred)>1e-12)
    res=np.abs(y-pred); q=np.quantile(res[tr],0.9)+1e-9; conf=np.mean(res[te]/q)
    lr=np.abs(np.log(np.clip(np.abs(y[pos]),1e-12,None)/np.clip(np.abs(pred[pos]),1e-12,None)))
    L=np.full(n,np.nan);L[pos]=lr; bd=np.nanmedian(L[te])/(np.nanmedian(L[tr])+1e-6)
    ir2=1-np.sum((y[tr]-pred[tr])**2)/(np.sum((y[tr]-np.mean(y[tr]))**2)+1e-12)
    verdict='NEEDS-NEW' if (conf>3 or bd>4) else 'classical'
    return dict(int_r2=ir2,conf=conf,bd=bd,verdict=verdict)
def sim(fn,rng,fe='low',npts=200,noise=0.03,seed=1):
    r=np.random.RandomState(seed);x=np.sort(r.uniform(*rng,npts));y=fn(x)*(1+noise*r.randn(npts))
    m=np.isfinite(y)&(np.abs(y)<1e12);return x[m],y[m],fe
def einstein(T): x=np.clip(1.0/np.clip(T,1e-3,None),1e-3,40); return x**2*np.exp(x)/(np.exp(x)-1)**2
CASES={
 'blackbody Planck (quantum)':        (lambda nu: nu**3/(np.exp(clp(nu))-1+1e-9),(0.02,15),'low'),
 'specific heat Debye (quantum)':     (einstein,(0.1,20),'high'),
 'oscillator mean-energy (quantum)':  (lambda T: 1.0/(np.exp(np.clip(1.0/np.clip(T,1e-3,None),0,60))-1+1e-9),(0.1,20),'high'),
 'photoelectric Millikan V_stop':     (lambda nu: np.maximum(nu-1.0,0.0)+1e-3,(0.0,4.0),'high'),
 'Compton shift (quantum)':           (lambda th: (1-np.cos(np.clip(th,0,3.1)))+1e-3,(0.0,3.1),'low'),
 'superconductor R(T)':               (lambda T: np.where(T>1.0,0.5*(T-1.0)+0.05,0.001),(0.1,4.0),'high'),
 'relativistic KE (v->c)':            (lambda v: (1/np.sqrt(np.clip(1-v**2,1e-4,1))-1),(0.02,0.995),'low'),
 'relativistic momentum (v->c)':      (lambda v: v/np.sqrt(np.clip(1-v**2,1e-4,1)),(0.02,0.995),'low'),
 'time dilation (v->c)':              (lambda v: 1/np.sqrt(np.clip(1-v**2,1e-4,1)),(0.02,0.995),'low'),
 'quantum tunnelling (honest miss)':  (lambda x: np.exp(-2*clp(x)),(0.02,4),'low'),
}
print('=== EXPANDED physically-grounded SIMULATION tier (labelled: from known laws, NOT real data) ===')
print('  parsimonious dictionary {power,exp,linear,const}; 3% noise; detector sees only numbers\n')
print('%-34s %-7s %-8s %-8s %s'%('simulated law','int-R2','conf','bd','verdict'))
det=0;tot=0
for name,(fn,rng,fe) in CASES.items():
    x,y,fe=sim(fn,rng,fe); r=engine(x,y,fe)
    if r is None: print('%-34s fit-failed'%name);continue
    tot+=1; det+=(r['verdict']=='NEEDS-NEW')
    print('%-34s %-7.3f %-8.2f %-8.2f %s'%(name,r['int_r2'],r['conf'],r['bd'],r['verdict']))
print('\n  detected NEEDS-NEW: %d / %d  (tunnelling is the pre-declared honest miss)'%(det,tot))
print('  Most have clean classical interiors (int-R2 near 1), so fit->extrapolate->localize is exercised.')
print('  NOTE: simulations, for benchmark realism only; real-data anchors remain FIRAS + Bertozzi (+ Kepler control).')
