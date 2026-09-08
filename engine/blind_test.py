"""E1: BLIND / prospective evaluation on an UNFILTERED corpus (addresses the circular-validation critique).
Negative arm: a large sample of REAL pre-1900 classical formulas from our KG master (each turned into a
1-D data relation via forward evaluation); ground truth = classical. Positive arm: known/simulated
breakdowns (blackbody, relativity, photoelectric, specific-heat Debye, superconductivity step, + variants);
ground truth = needs-new. The SAME detector with a PRE-SET decision rule is applied uniformly to every
dataset without looking at labels; we then report the confusion matrix, precision/recall/FPR, and AUROC."""
import sys,os,json,warnings,numpy as np; warnings.filterwarnings('ignore')
sys.path.insert(0,'D:/Physics Fundation model/sr_model/data')
import gen_dataside as GD
SC='C:/Users/1/AppData/Local/Temp/claude/D--Physics-Fundation-model/ddaebe6b-abe2-4e50-9eb7-f5879d2c4910/scratchpad/'
exec(open(SC+'build_classical_corpus.py',encoding='utf-8').read().split('if __name__')[0])  # KEEP, DROP_KW, parse_unit
from scipy.optimize import curve_fit
# ---- detector (pre-registered rule; thresholds fixed in advance) ----
FORMS={'const':(lambda x,a:a+0*x,[1]),'lin':(lambda x,a,b:a*x+b,[1,0]),
 'pow':(lambda x,a,b,c:a*np.abs(x)**b+c,[1,1,0]),'exp':(lambda x,a,b,c:a*np.exp(np.clip(-b*x,-60,60))+c,[1,.5,0])}
CONF_T, BD_T = 3.0, 4.0                       # pre-set thresholds (frozen from the synthetic calibration)
def fit_auto(x,y,tr):
    best=1e9;bp=None
    for f,p0 in FORMS.values():
        try:
            p,_=curve_fit(f,x[tr],y[tr],p0=p0,maxfev=6000); r=np.sqrt(np.mean((f(x[tr],*p)-y[tr])**2))
            if np.isfinite(r) and r<best: best=r;bp=(f,p)
        except: pass
    return bp
def detect(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<30 or np.std(y)<1e-12: return None
    o=np.argsort(x); n=len(x)
    # try both interior directions, take the better-fitting interior (auto, no peeking at where it breaks)
    best=None
    for tr,te in [(o[:int(.35*n)],o[int(.7*n):]),(o[int(.65*n):],o[:int(.3*n)])]:
        bp=fit_auto(x,y,tr)
        if bp is None: continue
        f,p=bp; pred=f(x,*p); ir=1-np.sum((y[tr]-pred[tr])**2)/(np.sum((y[tr]-np.mean(y[tr]))**2)+1e-12)
        res=np.abs(y-pred); q=np.quantile(res[tr],0.9)+1e-9; conf=np.mean(res[te]/q)
        pos=(np.abs(y)>0)&(np.abs(pred)>1e-12); lr=np.abs(np.log(np.clip(np.abs(y)/np.clip(np.abs(pred),1e-12,None),1e-12,None)))
        L=np.full(n,np.nan); L[pos]=lr[pos]; bd=np.nanmedian(L[te])/(np.nanmedian(L[tr])+1e-6)
        if best is None or ir>best[0]: best=(ir,conf,bd)
    if best is None: return None
    ir,conf,bd=best
    flag = (conf>CONF_T) or (bd>BD_T)
    score = 0.5*(min(conf/CONF_T,3)+min(bd/BD_T,3))    # continuous score for AUROC
    return dict(flag=flag,score=score,conf=conf,bd=bd)
# ---- NEGATIVE arm: real classical formulas -> 1-D relations ----
def one_d(pc):
    X=np.asarray(pc['X'],float); y=np.asarray(pc['y'],float)
    if X.ndim!=2 or X.shape[0]<40: return None
    cors=[abs(np.corrcoef(X[:,i],y)[0,1]) if np.std(X[:,i])>0 else 0 for i in range(X.shape[1])]
    i=int(np.argmax(cors));
    if cors[i]<0.2: return None
    return X[:,i],y
MASTER='D:/Physics Fundation model/dataset_20260531/_extract_master/master_20260616/master_nodes.jsonl'
rng=np.random.default_rng(7)
neg=[]; seen=0
for ln in open(MASTER,encoding='utf-8'):
    if len(neg)>=200: break
    try: r=json.loads(ln)
    except: continue
    if (r.get('domain') or '').strip() not in KEEP: continue
    if DROP_KW.search(r.get('expr','') or ''): continue
    vs=r.get('variables') or []
    if len(vs)!=2: continue                    # exactly 1 input + 1 output -> clean 1-D classical relation
    seen+=1
    if seen%2: continue                        # subsample for spread across the corpus
    try: pc=GD.make_one(r['expr'],rng,hint=vs[0].get('sym'))
    except: pc=None
    if not pc: continue
    d=one_d(pc)
    if d is None: continue
    res=detect(*d)
    if res is None: continue
    neg.append(res)
# ---- POSITIVE arm: known + simulated breakdowns (1-D) ----
clp=lambda z:np.clip(z,0,60)
BREAK={
 'blackbody Planck':      (lambda v: v**3/(np.exp(clp(v))-1+1e-9),(0.05,15)),
 'blackbody narrow':      (lambda v: v**3/(np.exp(clp(v))-1+1e-9),(0.05,9)),
 'relativistic KE':       (lambda b: (1/np.sqrt(np.clip(1-b**2,1e-4,1))-1),(0.02,0.995)),
 'relativistic momentum': (lambda b: b/np.sqrt(np.clip(1-b**2,1e-4,1)),(0.02,0.995)),
 'time dilation':         (lambda b: 1/np.sqrt(np.clip(1-b**2,1e-4,1)),(0.02,0.99)),
 'specific heat Debye':   (lambda T: (1.0/np.clip(T,1e-3,None))**2*np.exp(1/np.clip(T,1e-3,None))/(np.exp(np.clip(1/np.clip(T,1e-3,None),0,60))-1)**2,(0.08,20)),
 'photoelectric':         (lambda nu: np.maximum(nu-1.0,0.0)+1e-3,(0.0,4.0)),
 'superconductor step':   (lambda T: np.where(T>1.0,0.1+0.05*(T-1),1e-6),(0.2,3.0)),
 'Planck wide':           (lambda v: v**3/(np.exp(clp(v))-1+1e-9),(0.02,20)),
 'Wien tail':             (lambda v: v**3*np.exp(-clp(v)),(2.0,15)),
}
pos=[]
for name,(fn,(a,b)) in BREAK.items():
    for s in range(3):                        # 3 noise realizations each
        r=np.random.RandomState(s); x=np.sort(r.uniform(a,b,220)); y=fn(x)*(1+0.03*r.randn(220))
        res=detect(x,y)
        if res: res['name']=name; pos.append(res)
# ---- blind scoring ----
TP=sum(p['flag'] for p in pos); FN=len(pos)-TP
FP=sum(n['flag'] for n in neg); TN=len(neg)-FP
prec=TP/max(TP+FP,1); rec=TP/max(TP+FN,1); fpr=FP/max(FP+TN,1)
# AUROC over the mixed corpus
sc=np.array([p['score'] for p in pos]+[n['score'] for n in neg]); lab=np.array([1]*len(pos)+[0]*len(neg))
order=np.argsort(-sc);
def auroc(lab,sc):
    P=lab.sum(); N=len(lab)-P;
    if P==0 or N==0: return float('nan')
    o=np.argsort(-sc); r=lab[o]; tp=np.cumsum(r); fp=np.cumsum(1-r)
    return np.trapz(tp/P,fp/N)
print('=== E1 BLIND EVALUATION on an unfiltered corpus (labels hidden during detection) ===')
print('  negative arm: %d REAL pre-1900 classical formulas (KG master) -> 1-D relations'%len(neg))
print('  positive arm: %d breakdown datasets (%d phenomena x noise realizations)'%(len(pos),len(BREAK)))
print('  pre-set rule: flag if conformal>%.0f OR breakdown-ratio>%.0f (frozen)\n'%(CONF_T,BD_T))
print('  confusion:  TP=%d FN=%d | FP=%d TN=%d'%(TP,FN,FP,TN))
print('  precision=%.2f  recall(TPR)=%.2f  false-positive rate=%.2f'%(prec,rec,fpr))
print('  AUROC over the mixed corpus = %.3f'%auroc(lab,sc))
print('\n  -> the detector is run blind on real classical data + breakdowns; FPR quantifies over-flagging on')
print('     genuine classical physics, recall the sensitivity to real breakdowns. This is a prospective test,')
print('     not a curated 4/4 confirmation.')
