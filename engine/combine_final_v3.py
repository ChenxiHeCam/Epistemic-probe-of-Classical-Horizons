"""v3 FULLY leakage-free engine. Fix of the probe leak that keeps discriminative power:
the learned component's supervision is CLASSICAL vs GENERIC-DEFORMED-CLASSICAL. Negatives are built by
generic surgeries on pre-1900 classical relations (splice two classical laws, saturate beyond a point,
step change, kink) — the abstract notion 'the law changes somewhere', definable in 1900 with no reference
to any post-1900 physics. No post-1900 law enters training, calibration, or thresholds anywhere.
Also fixes the small-n calibration mismatch: tiny datasets are percentiled against size-matched
classical references. Post-1900-derived cases appear only as the held-out evaluation."""
import sys,os,json,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from models.encoders import DataEncoder
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
SC=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data','')
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
# ---------- classical clouds + GENERIC deformations (1900-legitimate supervision) ----------
D=np.load(SC+'classical_pointclouds.npz',allow_pickle=True); Xc=D['X']
rng=np.random.default_rng(11)
def deform(x,y,r):
    """generic 'law changes somewhere' surgeries, abrupt AND smooth; no post-1900 law referenced."""
    x=x.copy(); y=y.copy(); o=np.argsort(x); x,y=x[o],y[o]; n=len(x)
    kind=r.integers(0,6); j0=int(r.uniform(0.45,0.75)*n); scale=np.std(y)+1e-9; x0=x[j0]
    if kind==0:   # saturation: beyond x0 the response freezes
        y[j0:]=y[j0]+0.05*scale*r.standard_normal(n-j0)
    elif kind==1: # step: sudden offset drop/jump
        y[j0:]=y[j0:]-np.sign(y[j0]+1e-9)*r.uniform(1.5,4.0)*scale
    elif kind==2: # kink: slope flips/changes sharply beyond x0
        slope=(y[j0]-y[max(j0-5,0)])/(x[j0]-x[max(j0-5,0)]+1e-12)
        y[j0:]=y[j0]-r.uniform(1.0,3.0)*slope*(x[j0:]-x[j0])
    elif kind==3: # splice: graft a DIFFERENT classical relation's tail (rescaled)
        k=r.integers(0,len(Xc)); other=Xc[k]; oy=other[np.argsort(other[:,0]),1]
        tail=oy[:n-j0]; tail=(tail-tail.mean())/(np.std(tail)+1e-9)*scale+y[j0]
        y[j0:]=tail
    elif kind==4: # smooth rolloff: the law gradually gives way beyond x0 (smooth turnover)
        w=r.uniform(0.1,0.5)*(x[-1]-x0+1e-9)
        y=y/(1.0+((np.maximum(x-x0,0))/w)**r.uniform(1.5,3.0))
    else:         # smooth crossover: sigmoid blend into a different classical relation
        k=r.integers(0,len(Xc)); other=Xc[k]; oy=other[np.argsort(other[:,0]),1][:n]
        if len(oy)<n: oy=np.pad(oy,(0,n-len(oy)),mode='edge')
        oy=(oy-oy.mean())/(np.std(oy)+1e-9)*scale+np.mean(y)
        w=r.uniform(0.05,0.25)*(x[-1]-x[0]); s=1/(1+np.exp(-(x-x0)/ (w+1e-9)))
        y=(1-s)*y+s*oy
    return x,y
idx=rng.choice(len(Xc),3000,replace=False)
Zp=[];Zn=[]
for i in idx:
    c=Xc[i]; z=zd(c[:,0],c[:,1])
    if z is not None: Zp.append(z)
    xr,yr=deform(c[:,0],c[:,1],rng); z2=zd(xr,yr)
    if z2 is not None: Zn.append(z2)
Zp=np.array(Zp);Zn=np.array(Zn)
print('probe training: classical %d vs generic-deformed %d (no post-1900 anywhere)'%(len(Zp),len(Zn)))
X=np.vstack([Zp,Zn]); Y=np.array([0]*len(Zp)+[1]*len(Zn))
clf=LogisticRegression(max_iter=3000,C=0.1).fit(X,Y)
# ---------- statistical component ----------
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
# ---------- classical-only, SIZE-MATCHED calibration ----------
cal_idx=rng.choice(len(Xc),400,replace=False)
CAL_FULL=[]; CAL_SMALL=[]
for i in cal_idx:
    c=Xc[i]; ss=stat_score(c[:,0],c[:,1])
    if ss is not None: CAL_FULL.append(ss)
    sel=rng.choice(len(c),8,replace=False); ss2=stat_score(c[sel,0],c[sel,1])   # size-matched (n=8) reference
    if ss2 is not None: CAL_SMALL.append(ss2)
CAL_FULL=np.array(CAL_FULL); CAL_SMALL=np.array(CAL_SMALL)
def pctl(v,ref): return float((ref<v).mean())
# combination weight ALPHA chosen on a held-out classical-vs-deformed split (no post-1900 anywhere)
val_idx=rng.choice(len(Xc),300,replace=False)
vs_s=[];vs_p=[];vs_y=[]
for i in val_idx:
    c=Xc[i]
    for lab2,(xx2,yy2) in [(0,(c[:,0],c[:,1])),(1,deform(c[:,0],c[:,1],rng))]:
        ss=stat_score(xx2,yy2); z=zd(xx2,yy2)
        if ss is None or z is None: continue
        vs_s.append(pctl(ss,CAL_FULL)); vs_p.append(clf.predict_proba(z[None])[0,1]); vs_y.append(lab2)
vs_s=np.array(vs_s);vs_p=np.array(vs_p);vs_y=np.array(vs_y)
best=(0.5,0)
for a in np.linspace(0,1,21):
    auc=roc_auc_score(vs_y,a*vs_p+(1-a)*vs_s)
    if auc>best[1]: best=(a,auc)
ALPHA=best[0]; print('combination weight alpha (probe share) = %.2f  (val AUROC %.3f on classical-vs-deformed)'%(ALPHA,best[1]))
def comb_score(x,y):
    x=np.asarray(x,float); n=len(x)
    ss=stat_score(x,y)
    if ss is None: return None
    ps=pctl(ss,CAL_SMALL if n<20 else CAL_FULL)
    z=zd(x,y)
    if z is None: return ps
    pl=clf.predict_proba(z[None])[0,1]
    return ALPHA*pl+(1-ALPHA)*ps
def learn_score(x,y):
    z=zd(x,y); return clf.predict_proba(z[None])[0,1] if z is not None else None
# ---------- held-out evaluation (post-1900-derived = TEST ONLY) ----------
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
Sst=[];Sln=[];Scb=[];Yl=[]
for lab,fns in [(0,CLASS),(1,BREAK)]:
    for fn in fns:
        for s in range(3):
            r=np.random.RandomState(s); xx=np.sort(r.uniform(0.1,6,200)); yy=fn(xx)*(1+0.02*r.randn(200))
            ss=stat_score(xx,yy); ls=learn_score(xx,yy); cs=comb_score(xx,yy)
            if None in (ss,ls,cs): continue
            Sst.append(pctl(ss,CAL_FULL)); Sln.append(ls); Scb.append(cs); Yl.append(lab)
Yl=np.array(Yl)
print('=== v3 FULLY LEAKAGE-FREE engine (generic-deformation probe) — held-out suite n=%d ==='%len(Yl))
print('  statistical               AUROC = %.3f'%roc_auc_score(Yl,Sst))
print('  learned (deform-probe)    AUROC = %.3f'%roc_auc_score(Yl,Sln))
print('  COMBINED                  AUROC = %.3f'%roc_auc_score(Yl,Scb))
# ---------- blind corpus ----------
# 200 real pre-1900 classical relations, pre-generated from the source knowledge base
# with a fixed seed and shipped as data/classical_relations_1d.npz
NPZ=np.load(SC+'classical_relations_1d.npz',allow_pickle=True)
neg=[(np.asarray(_x,float),np.asarray(_y,float)) for _x,_y in zip(NPZ['x'],NPZ['y'])]
POS=[(BREAK[0],(0.05,15)),(BREAK[0],(0.05,9)),(BREAK[1],(0.02,0.995)),(BREAK[2],(0.02,0.995)),
 (lambda b:1/np.sqrt(np.clip(1-b**2,1e-4,1)),(0.02,0.99)),(BREAK[3],(0.08,20)),(BREAK[4],(0.0,4.0)),
 (BREAK[5],(0.2,3.0)),(BREAK[0],(0.02,20)),(BREAK[6],(2.0,15))]
sc=[];lab=[]
for fn,(a,b) in POS:
    for s in range(3):
        r=np.random.RandomState(s); x=np.sort(r.uniform(a,b,220)); y=fn(x)*(1+0.03*r.randn(220))
        v=comb_score(x,y)
        if v is not None: sc.append(v); lab.append(1)
for x,y in neg:
    v=comb_score(x,y)
    if v is not None: sc.append(v); lab.append(0)
print('\n=== BLIND corpus, v3 combined ===  AUROC = %.3f (neg=%d, pos=%d)'%(roc_auc_score(np.array(lab),np.array(sc)),len(neg),sum(lab)))
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
boyle=np.array(json.load(open(SC+'boyle_1662.json')))          # Boyle 1662, 25 real P-V points (public domain)
gm_a=np.array([421.8,671.1,1070.4,1882.7])                     # Galilean moons: semi-major axis (10^3 km)
gm_P=np.array([1.769,3.551,7.155,16.689])                      # periods (days) — real astronomical constants
print('\n=== real cases, v3 combined ===')
for name,x,y,tr in [('FIRAS',nu,I,'break'),('specific heat Cu',Tc,Cp,'break'),('Bertozzi',KEb,b2,'break'),('Onnes',To,Ro,'break'),('Kepler control',ak,Pk,'classical'),
                    ('Boyle 1662 control',boyle[:,0],boyle[:,1],'classical'),('Galilean moons control',gm_a,gm_P,'classical')]:
    v=comb_score(x,y); mode='combined' if len(x)>=20 else 'stat-only(size-matched cal)'
    if v is None:
        print('  %-18s score=n/a   (below the minimum point count for both components)  expected=%s'%(name,tr)); continue
    print('  %-18s score=%.2f  (%s)  expected=%s'%(name,v,mode,tr))
