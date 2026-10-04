"""FULLY leakage-free combined engine (v2), removing the probe-supervision leak of combine_final.py:
the learned component is now ONE-CLASS. The frozen 1900-pure encoder maps a point cloud to a latent;
the reference distribution is the latents of REAL pre-1900 classical relations only; the learned score is
the kNN distance to that classical reference (no post-1900 supervision anywhere in the detector).
Calibration percentiles for BOTH components are computed against classical-only references.
Post-1900-derived cases appear only as the evaluation test set (ground truth for AUROC), never in fitting."""
import sys,os,json,warnings; warnings.filterwarnings('ignore')
import legacy_paths  # repository-relative paths and external inputs
legacy_paths.add_repo_root()
legacy_paths.add_gen_dataside()
import numpy as np, torch
from models.encoders import DataEncoder
from scipy.optimize import curve_fit
SC=legacy_paths.SC
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
    z=torch.nn.functional.normalize(z,dim=-1)[0].numpy()
    return z if np.all(np.isfinite(z)) else None
# ---------- classical reference latents (ONE-CLASS training = classical only) ----------
D=np.load(SC+'classical_pointclouds.npz',allow_pickle=True); Xc=D['X']
rng=np.random.default_rng(11); idx=rng.choice(len(Xc),2000,replace=False)
REF=[]
with torch.no_grad():
    for i in idx:
        c=Xc[i]; z=zd(c[:,0],c[:,1])
        if z is not None: REF.append(z)
REF=np.array(REF); print('classical reference latents:',REF.shape)
def knn_score(z,k=10):
    d=1-REF@z                                    # cosine distance to classical reference
    return float(np.sort(d)[:k].mean())
# ---------- statistical component (unchanged) ----------
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
# ---------- classical-only calibration references for BOTH components ----------
CAL_STAT=[]; CAL_KNN=[]
for i in rng.choice(len(Xc),400,replace=False):
    c=Xc[i]; ss=stat_score(c[:,0],c[:,1])
    if ss is not None: CAL_STAT.append(ss)
    z=zd(c[:,0],c[:,1])
    if z is not None: CAL_KNN.append(knn_score(z))
CAL_STAT=np.array(CAL_STAT); CAL_KNN=np.array(CAL_KNN)
def pctl(v,ref): return float((ref<v).mean())
def comb_score(x,y):
    ss=stat_score(x,y)
    if ss is None: return None
    ps=pctl(ss,CAL_STAT); z=zd(x,y)
    if z is None: return ps
    pl=pctl(knn_score(z),CAL_KNN)
    return 0.5*(ps+pl)
def learn_score(x,y):
    z=zd(x,y)
    return pctl(knn_score(z),CAL_KNN) if z is not None else None
# ---------- evaluation suite (post-1900-derived cases = TEST ONLY) ----------
clp=lambda z:np.clip(z,0,60)
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
from sklearn.metrics import roc_auc_score
Sst=[];Sln=[];Scb=[];Y=[]
for lab,fns in [(0,CLASS),(1,BREAK)]:
    for fn in fns:
        for s in range(3):
            r=np.random.RandomState(s); xx=np.sort(r.uniform(0.1,6,200)); yy=fn(xx)*(1+0.02*r.randn(200))
            ss=stat_score(xx,yy); ls=learn_score(xx,yy); cs=comb_score(xx,yy)
            if None in (ss,ls,cs): continue
            Sst.append(pctl(ss,CAL_STAT)); Sln.append(ls); Scb.append(cs); Y.append(lab)
Y=np.array(Y)
print('=== v2 FULLY LEAKAGE-FREE engine (one-class learned component) — evaluation suite n=%d ==='%len(Y))
print('  statistical (classical-calibrated)  AUROC = %.3f'%roc_auc_score(Y,Sst))
print('  learned ONE-CLASS (kNN to classical) AUROC = %.3f'%roc_auc_score(Y,Sln))
print('  COMBINED                             AUROC = %.3f'%roc_auc_score(Y,Scb))
# ---------- blind corpus ----------
exec(open(legacy_paths.CORPUS_BUILDER,encoding='utf-8').read().split('if __name__')[0])
import gen_dataside as GD
MASTER=legacy_paths.master_graph()
rng2=np.random.default_rng(7); neg=[]; seen=0
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
    try: pc=GD.make_one(r['expr'],rng2,hint=vs[0].get('sym'))
    except: pc=None
    if not pc: continue
    X=np.asarray(pc['X'],float); yv=np.asarray(pc['y'],float)
    if X.ndim!=2 or X.shape[0]<40: continue
    cors=[abs(np.corrcoef(X[:,i],yv)[0,1]) if np.std(X[:,i])>0 else 0 for i in range(X.shape[1])]
    i=int(np.argmax(cors))
    if cors[i]<0.2: continue
    neg.append((X[:,i],yv))
BL=[(fn,(a,b)) for fn,(a,b) in [
 (BREAK[0],(0.05,15)),(BREAK[0],(0.05,9)),(BREAK[1],(0.02,0.995)),(BREAK[2],(0.02,0.995)),
 (lambda b:1/np.sqrt(np.clip(1-b**2,1e-4,1)),(0.02,0.99)),(BREAK[3],(0.08,20)),(BREAK[4],(0.0,4.0)),
 (BREAK[5],(0.2,3.0)),(BREAK[0],(0.02,20)),(BREAK[6],(2.0,15))]]
sc=[];lab=[]
for fn,(a,b) in BL:
    for s in range(3):
        r=np.random.RandomState(s); x=np.sort(r.uniform(a,b,220)); y=fn(x)*(1+0.03*r.randn(220))
        v=comb_score(x,y)
        if v is not None: sc.append(v); lab.append(1)
for x,y in neg:
    v=comb_score(x,y)
    if v is not None: sc.append(v); lab.append(0)
print('\n=== BLIND corpus, v2 combined ===')
print('  AUROC = %.3f  (n_neg=%d real classical relations, n_pos=%d)'%(roc_auc_score(np.array(lab),np.array(sc)),len(neg),sum(lab)))
# ---------- real cases ----------
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
print('\n=== real cases, v2 combined ===')
for name,x,y,tr in [('FIRAS',nu,I,'break'),('specific heat Cu',Tc,Cp,'break'),('Bertozzi',KEb,b2,'break'),('Onnes',To,Ro,'break'),('Kepler control',ak,Pk,'classical')]:
    v=comb_score(x,y); mode='combined' if len(x)>=20 else 'stat-only'
    print('  %-18s score=%.2f  (%s)  expected=%s'%(name,v,mode,tr))
