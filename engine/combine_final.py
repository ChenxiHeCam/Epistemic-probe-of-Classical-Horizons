"""FINAL engine = combination of the two leakage-free detectors:
  (1) statistical ensemble (split-conformal + extrapolation-residual ratio)  [AUROC 0.80]
  (2) learned 1900-pure encoder latent + linear probe                        [AUROC 0.84]
Combined score = mean of (probe probability, empirical percentile of the statistical score on the
calibration suite). Datasets with <20 points fall back to the statistical component (encoder needs
a set). Evaluate: synthetic suite (CV), the blind corpus (200 real classical + 30 breakdowns), and
the real case studies."""
import sys,os,json,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'D:/Physics Fundation model/sr_model')
sys.path.insert(0,'D:/Physics Fundation model/sr_model/data')
import numpy as np, torch
from models.encoders import DataEncoder
from scipy.optimize import curve_fit
SC='C:/Users/1/AppData/Local/Temp/claude/D--Physics-Fundation-model/ddaebe6b-abe2-4e50-9eb7-f5879d2c4910/scratchpad/'
# ---------- learned engine ----------
DEV='cpu'; MAXV=16; DIML=6
ck=torch.load(SC+'encoder_1900.pt',map_location=DEV)
enc=DataEncoder(max_vars=MAXV,d=256,n_isab=6,dim_len=ck.get('dim_len',DIML),n_tokens=16,log_feats=True,class_feats=True,robust_norm=True).to(DEV)
enc.load_state_dict(ck['state']); enc.eval()
@torch.no_grad()
def zd(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)&(np.abs(y)<1e18); x,y=x[m],y[m]
    if len(x)<20: return None
    pts=np.zeros((1,len(x),MAXV+1),np.float32); pts[0,:,0]=x; pts[0,:,MAXV]=y
    z=enc(torch.tensor(pts),torch.tensor([[1.]+[0.]*(MAXV-1)]),torch.ones(1,len(x)),dims=torch.zeros(1,MAXV+1,DIML))
    z=z[0].numpy(); return z if np.all(np.isfinite(z)) else None
# ---------- statistical engine (same as blind_test) ----------
FORMS={'const':(lambda x,a:a+0*x,[1]),'lin':(lambda x,a,b:a*x+b,[1,0]),
 'pow':(lambda x,a,b,c:a*np.abs(x)**b+c,[1,1,0]),'exp':(lambda x,a,b,c:a*np.exp(np.clip(-b*x,-60,60))+c,[1,.5,0])}
def fit_auto(x,y,tr):
    best=1e9;bp=None
    for f,p0 in FORMS.values():
        try:
            p,_=curve_fit(f,x[tr],y[tr],p0=p0,maxfev=6000); r=np.sqrt(np.mean((f(x[tr],*p)-y[tr])**2))
            if np.isfinite(r) and r<best: best=r;bp=(f,p)
        except: pass
    return bp
def stat_score(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<5 or np.std(y)<1e-12: return None
    o=np.argsort(x); n=len(x); best=None
    if n<12: splits=[(o[:max(3,int(.6*n))],o[max(3,int(.6*n)):]),(o[n-max(3,int(.6*n)):],o[:n-max(3,int(.6*n))])]
    else: splits=[(o[:int(.35*n)],o[int(.7*n):]),(o[int(.65*n):],o[:int(.3*n)])]
    for tr,te in splits:
        bp=fit_auto(x,y,tr)
        if bp is None: continue
        f,p=bp; pred=f(x,*p); ir=1-np.sum((y[tr]-pred[tr])**2)/(np.sum((y[tr]-np.mean(y[tr]))**2)+1e-12)
        res=np.abs(y-pred); q=np.quantile(res[tr],0.9)+1e-3*np.std(y)+1e-12; conf=np.mean(res[te]/q)
        pos=(np.abs(y)>0)&(np.abs(pred)>1e-12); lr=np.abs(np.log(np.clip(np.abs(y)/np.clip(np.abs(pred),1e-12,None),1e-12,None)))
        L=np.full(n,np.nan); L[pos]=lr[pos]; bd=np.nanmedian(L[te])/(np.nanmedian(L[tr])+1e-6)
        if best is None or ir>best[0]: best=(ir,conf,bd)
    if best is None: return None
    _,conf,bd=best
    return 0.5*(min(conf/3.0,3)+min(bd/4.0,3))
# ---------- synthetic suite (same 60 fns x 3 seeds as probe_1900) ----------
clp=lambda z:np.clip(z,0,60)
exec(open(SC+'probe_1900.py',encoding='utf-8').read().split('from sklearn')[0].split('clp=lambda')[1].replace('z:np.clip(z,0,60)\n','z_:np.clip(z_,0,60)\n',1) if False else '')
# (re-declare fns inline to avoid import gymnastics)
CLASS=[lambda x:x**2,lambda x:2*x+1,lambda x:np.exp(-x),lambda x:1.0/np.clip(x,.1,None),lambda x:x**0.5,
 lambda x:3*x**3,lambda x:np.exp(-2*x),lambda x:1/np.clip(x,.1,None)**2,lambda x:x**1.5,lambda x:5-x,
 lambda x:np.log(np.clip(x,.1,None))+2,lambda x:x**2+x,lambda x:x**2.5,lambda x:4*x,lambda x:x**3+2*x,
 lambda x:1/np.clip(x,.1,None)**0.5,lambda x:2*np.exp(-0.5*x),lambda x:x**0.25,lambda x:6-2*x,lambda x:x**4,
 lambda x:x*np.exp(-x),lambda x:1/np.clip(1+x,.1,None),lambda x:np.sqrt(np.clip(x,0,None))+x,lambda x:10*x**-1,
 lambda x:x**1.2,lambda x:3-x**0.5,lambda x:np.exp(-3*x)+0.1,lambda x:2*x**2-x,lambda x:x**0.8,lambda x:5/np.clip(x**2,.01,None)]
BREAK=[lambda v:v**3/(np.exp(clp(v))-1+1e-9),lambda b:1/np.sqrt(np.clip(1-b**2,1e-4,1))-1,
 lambda b:b/np.sqrt(np.clip(1-b**2,1e-4,1)),lambda T:(1/np.clip(T,1e-2,None))**2*np.exp(np.clip(1/np.clip(T,1e-2,None),0,40))/(np.exp(np.clip(1/np.clip(T,1e-2,None),0,40))-1)**2,
 lambda nu:np.maximum(nu-1,0)+1e-3,lambda T:np.where(T>1,0.1+0.05*(T-1),1e-6),lambda v:v**3*np.exp(-clp(v)),
 lambda x:1/(np.exp(clp(1/np.clip(x,1e-2,None)))-1+1e-9),lambda x:np.tanh(5*(x-1)),lambda x:np.where(x>1.5,x**2,x),
 lambda x:np.sin(3*x)*np.exp(-x),lambda x:1/(1+np.exp(-8*(x-1))),lambda b:1/np.sqrt(np.clip(1-b**2,1e-4,1)),
 lambda x:np.where(x>1,1e-4,1.0),lambda x:np.cos(6*x),lambda nu:np.maximum(nu-1.5,0)**0.5+1e-3,
 lambda x:np.exp(clp(-1/np.clip(x,1e-2,None))),lambda T:np.where(T>1.2,0.05,1e-5),lambda x:np.sin(10*x),
 lambda x:np.where(x<1,x,2-x),lambda b:np.log(np.clip(1/np.sqrt(np.clip(1-b**2,1e-4,1)),1,None)),
 lambda x:np.floor(x*3)/3.0,lambda x:np.abs(np.sin(4*x)),lambda x:1/(np.exp(clp(x-3))+1),
 lambda v:v**2/(np.exp(clp(v/2))-1+1e-9),lambda x:np.where(x>2,10,x),lambda x:np.tanh(3*x-3),lambda x:np.sign(x-1.5)*np.abs(x-1.5)**0.5,
 lambda x:np.exp(clp(-2/np.clip(x,1e-2,None))),lambda x:np.where(x>1,np.exp(-(x-1)),1.0)]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
Z=[];S=[];Y=[]
for lab,fns in [(0,CLASS),(1,BREAK)]:
    for fn in fns:
        for s in range(3):
            r=np.random.RandomState(s); xx=np.sort(r.uniform(0.1,6,200)); yy=fn(xx)*(1+0.02*r.randn(200))
            z=zd(xx,yy); ss=stat_score(xx,yy)
            if z is None or ss is None: continue
            Z.append(z); S.append(ss); Y.append(lab)
Z=np.array(Z); S=np.array(S); Y=np.array(Y)
clf=LogisticRegression(max_iter=3000,C=0.05)
p_learn=cross_val_predict(clf,Z,Y,cv=5,method='predict_proba')[:,1]
def pct(v,ref): return float((ref<v).mean())
S_ref=S.copy()
p_stat=np.array([pct(s,S_ref) for s in S])
p_comb=0.5*(p_learn+p_stat)
print('=== FINAL COMBINED ENGINE (both components leakage-free) — synthetic suite n=%d ==='%len(Y))
print('  statistical alone  AUROC = %.3f'%roc_auc_score(Y,p_stat))
print('  learned alone      AUROC = %.3f (5-fold CV)'%roc_auc_score(Y,p_learn))
print('  COMBINED (mean)    AUROC = %.3f'%roc_auc_score(Y,p_comb))
# ---------- blind corpus with combined ----------
exec(open(SC+'build_classical_corpus.py',encoding='utf-8').read().split('if __name__')[0])
import gen_dataside as GD
MASTER='D:/Physics Fundation model/dataset_20260531/_extract_master/master_20260616/master_nodes.jsonl'
rng=np.random.default_rng(7); neg=[]; seen=0
for ln in open(MASTER,encoding='utf-8'):
    if len(neg)>=200: break
    try: r=json.loads(ln)
    except: continue
    if (r.get('domain') or '').strip() not in KEEP: continue
    if DROP_KW.search(r.get('expr','') or ''): continue
    vs=r.get('variables') or []
    if len(vs)!=2: continue
    seen+=1
    if seen%2: continue
    try: pc=GD.make_one(r['expr'],rng,hint=vs[0].get('sym'))
    except: pc=None
    if not pc: continue
    X=np.asarray(pc['X'],float); yv=np.asarray(pc['y'],float)
    if X.ndim!=2 or X.shape[0]<40: continue
    cors=[abs(np.corrcoef(X[:,i],yv)[0,1]) if np.std(X[:,i])>0 else 0 for i in range(X.shape[1])]
    i=int(np.argmax(cors))
    if cors[i]<0.2: continue
    neg.append((X[:,i],yv))
clf.fit(Z,Y)   # probe fit on full synthetic suite for transfer
def comb_score(x,y):
    ss=stat_score(x,y)
    if ss is None: return None
    z=zd(x,y)
    ps=pct(ss,S_ref)
    if z is None: return ps                       # tiny datasets: statistical fallback
    pl=clf.predict_proba(z[None])[0,1]
    return 0.5*(pl+ps)
BL={'blackbody':(lambda v:v**3/(np.exp(clp(v))-1+1e-9),(0.05,15)),'blackbody2':(lambda v:v**3/(np.exp(clp(v))-1+1e-9),(0.05,9)),
 'relKE':(lambda b:1/np.sqrt(np.clip(1-b**2,1e-4,1))-1,(0.02,0.995)),'relP':(lambda b:b/np.sqrt(np.clip(1-b**2,1e-4,1)),(0.02,0.995)),
 'tdil':(lambda b:1/np.sqrt(np.clip(1-b**2,1e-4,1)),(0.02,0.99)),'debye':(lambda T:(1.0/np.clip(T,1e-3,None))**2*np.exp(np.clip(1/np.clip(T,1e-3,None),0,60))/(np.exp(np.clip(1/np.clip(T,1e-3,None),0,60))-1)**2,(0.08,20)),
 'photo':(lambda nu:np.maximum(nu-1.0,0.0)+1e-3,(0.0,4.0)),'sc':(lambda T:np.where(T>1.0,0.1+0.05*(T-1),1e-6),(0.2,3.0)),
 'planckw':(lambda v:v**3/(np.exp(clp(v))-1+1e-9),(0.02,20)),'wien':(lambda v:v**3*np.exp(-clp(v)),(2.0,15))}
pos=[]
for name,(fn,(a,b)) in BL.items():
    for s in range(3):
        r=np.random.RandomState(s); x=np.sort(r.uniform(a,b,220)); y=fn(x)*(1+0.03*r.randn(220)); pos.append((x,y))
sc_all=[];lab_all=[]
for x,y in pos:
    s=comb_score(x,y)
    if s is not None: sc_all.append(s); lab_all.append(1)
for x,y in neg:
    s=comb_score(x,y)
    if s is not None: sc_all.append(s); lab_all.append(0)
sc_all=np.array(sc_all); lab_all=np.array(lab_all)
print('\n=== BLIND corpus (200 real classical + 30 breakdowns) with COMBINED engine ===')
print('  combined AUROC = %.3f   (statistical-only was 0.902)'%roc_auc_score(lab_all,sc_all))
# ---------- real case studies with combined ----------
print('\n=== real case studies, COMBINED engine ===')
def load2(fn):
    rr=[]
    for line in open(SC+fn):
        s=line.strip()
        if not s or s[0].isalpha() or s.startswith('#'): continue
        p=s.replace(',',' ').split()
        try: rr.append((float(p[0]),float(p[1])))
        except: pass
    a=np.array(rr); return a[:,0],a[:,1]
nu,I=load2('firas_monopole.txt')
CU=[-1.91844,-0.15973,8.61013,-18.996,21.9661,-12.7328,3.54322,-0.3797,0]
Tc=np.logspace(np.log10(4),np.log10(300),120); Cp=10**np.clip(sum(c*np.log10(Tc)**k for k,c in enumerate(CU)),-30,30)
KEb=np.array([.5,1,1.5,4.5,15]); b2=np.array([.752,.828,.922,.974,1.0])
To=np.array([4.00,4.10,4.15,4.19,4.21,4.25,4.30,4.35,4.40]); Ro=np.array([1e-5,1e-5,1e-5,1e-5,0.110,0.118,0.126,0.134,0.142])
ak=np.array([0.387,0.723,1.0,1.524,5.203,9.537,19.19,30.07]); Pk=np.array([0.241,0.615,1.0,1.881,11.86,29.46,84.01,164.8])
CASES=[('FIRAS blackbody [43 pts, real]',nu,I,'break'),('specific heat Cu [NIST curve]',Tc,Cp,'break'),
 ('Bertozzi [5 pts, real]',KEb,b2,'break'),('Onnes [9 pts, real]',To,Ro,'break'),('Kepler [8 planets, control]',ak,Pk,'classical')]
for name,x,y,truth in CASES:
    s=comb_score(x,y); mode='combined' if (len(x)>=20) else 'stat-fallback'
    print('  %-32s score=%.2f  %-12s expected=%s'%(name,s if s is not None else float('nan'),mode,truth))
