"""REAL DATA #3 (3rd breakdown phenomenon: quantum thermodynamics), run through the SAME ensemble engine
as the rest of the paper. OFHC copper specific heat, NIST cryogenic reference (fit to measured data,
4-300 K). Classical Dulong-Petit predicts a constant molar heat capacity (clean classical interior at
high T); the real curve collapses at low T (Debye). Robustness across 4 real NIST materials appended."""
import numpy as np,json
from scipy.optimize import curve_fit
import warnings; warnings.filterwarnings('ignore')
# --- same engine as sim_historical2.py (parsimonious dict, conformal + breakdown ratio) ---
FORMS={'const':(lambda x,a:a+0*x,[1]),'lin':(lambda x,a,b:a*x+b,[1,0]),
 'pow':(lambda x,a,b,c:a*np.abs(x)**b+c,[1,1,0]),
 'exp':(lambda x,a,b,c:a*np.exp(np.clip(-b*x,-60,60))+c,[1,.5,0])}
def fit_auto(x,y,tr):
    best=1e9;bp=None
    for fm,(f,p0) in FORMS.items():
        try:
            p,_=curve_fit(f,x[tr],y[tr],p0=p0,maxfev=8000);r=np.sqrt(np.mean((f(x[tr],*p)-y[tr])**2))
            if np.isfinite(r) and r<best: best=r;bp=(f,p,r)
        except: pass
    return bp
def engine(x,y,interior='high'):
    o=np.argsort(x);n=len(x)
    tr=o[int(.65*n):] if interior=='high' else o[:int(.35*n)]
    te=o[:int(.30*n)] if interior=='high' else o[int(.70*n):]
    bp=fit_auto(x,y,tr); f,p,ir=bp; pred=f(x,*p)
    pos=(np.abs(y)>0)&(np.abs(pred)>1e-12)
    res=np.abs(y-pred); q=np.quantile(res[tr],0.9)+1e-9; conf=np.mean(res[te]/q)
    lr=np.abs(np.log(np.clip(np.abs(y[pos]),1e-12,None)/np.clip(np.abs(pred[pos]),1e-12,None)))
    L=np.full(n,np.nan);L[pos]=lr; bd=np.nanmedian(L[te])/(np.nanmedian(L[tr])+1e-6)
    ir2=1-np.sum((y[tr]-pred[tr])**2)/(np.sum((y[tr]-np.mean(y[tr]))**2)+1e-12)
    verdict='NEEDS-NEW' if (conf>3 or bd>4) else 'classical'
    return dict(int_r2=ir2,conf=conf,bd=bd,verdict=verdict,form=[k for k,v in FORMS.items() if v[0] is f][0])
# --- OFHC copper (NIST) ---
CU=[-1.91844,-0.15973,8.61013,-18.996,21.9661,-12.7328,3.54322,-0.3797,0]
def cp(coef,T): L=np.log10(T); return 10**np.clip(sum(c*L**k for k,c in enumerate(coef)),-30,30)
T=np.logspace(np.log10(4),np.log10(300),140); Cp=cp(CU,T)
r=engine(T,Cp,interior='high')
print('=== REAL DATA #3: OFHC copper specific heat (NIST reference, 4-300 K) via paper ensemble engine ===')
print('  classical best-fit form on high-T interior : %s'%r['form'])
print('  interior R^2 (clean classical plateau)     : %.4f'%r['int_r2'])
print('  conformal nonconformity (frontier)         : %.1f'%r['conf'])
print('  breakdown ratio (frontier/interior)        : %.1f'%r['bd'])
print('  verdict                                    : %s  (low-T Debye/quantum onset)'%r['verdict'])
print('  classical Dulong-Petit over-predicts at 4 K by %.0fx (data %.3f vs classical ~%.0f J/kg-K)'%(
      np.median(Cp[T>150])/cp(CU,np.array([4]))[0], cp(CU,np.array([4]))[0], np.median(Cp[T>150])))
print('  --> a clean classical interior on REAL data (unlike FIRAS): fit->extrapolate->localize fully exercised.\n')
# --- robustness across 4 clean NIST materials ---
rows=json.load(open('nist_cp_coef.json'))
print('=== robustness: same quantum-thermodynamics breakdown across real NIST materials ===')
print('%-20s %-8s %-8s %-8s %-8s %s'%('material','int-R2','conf','bd','Cp(300)','verdict'))
keep=[]
for name,coef,rng,*rest in rows:
    lo,hi=rng; Tm=np.logspace(np.log10(max(lo,4)),np.log10(hi),140); C=cp(coef,Tm)
    if not np.all(np.isfinite(C)) or C[-1]<=0 or C.max()/max(C[-1],1e-9)>5: continue  # drop numeric blowups
    rr=engine(Tm,C,interior='high')
    print('%-20s %-8.3f %-8.1f %-8.1f %-8.0f %s'%(name,rr['int_r2'],min(rr['conf'],9999),rr['bd'],C[-1],rr['verdict']))
    keep.append((name,rr['verdict']))
nd=sum(1 for _,v in keep if v=='NEEDS-NEW')
print('\n  %d/%d real NIST materials flagged NEEDS-NEW (classical Dulong-Petit -> low-T quantum), same engine.'%(nd,len(keep)))
print('  Real-data breakdown phenomena now: FIRAS (quantum radiation) + Bertozzi (relativity) + specific heat (quantum thermo).')
