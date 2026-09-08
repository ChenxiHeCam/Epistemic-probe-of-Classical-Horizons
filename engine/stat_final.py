"""Statistical component (leakage-free) final numbers on all real cases + 3 real controls,
with classical-only, size-matched calibration."""
import sys,json,warnings,numpy as np; warnings.filterwarnings('ignore')
from scipy.optimize import curve_fit
SC='C:/Users/1/AppData/Local/Temp/claude/D--Physics-Fundation-model/ddaebe6b-abe2-4e50-9eb7-f5879d2c4910/scratchpad/'
D=np.load(SC+'classical_pointclouds.npz',allow_pickle=True); Xc=D['X']
rng=np.random.default_rng(11)
POW=lambda x,a,b,c:a*np.abs(x)**b+c
EXP=lambda x,a,b,c:a*np.exp(np.clip(-b*x,-60,60))+c
FORMS=[(lambda x,a:a+0*x,[[1]]),(lambda x,a,b:a*x+b,[[1,0]]),
 (POW,[[1,1,0],[1,-1,0],[1,2,0],[1,0.5,0],[1,-2,0],[1,1.5,0]]),      # multi-start over exponents
 (EXP,[[1,.5,0],[1,2,0],[1,.05,0],[-1,.5,0]])]
def fit_auto(x,y,tr):
    best=1e9;bp=None
    for f,p0s in FORMS:
        for p0 in p0s:
            try:
                p,_=curve_fit(f,x[tr],y[tr],p0=p0,maxfev=6000); r=np.sqrt(np.mean((f(x[tr],*p)-y[tr])**2))
                if np.isfinite(r) and r<best: best=r;bp=(f,p)
            except: pass
    return bp
def stat_score(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<4 or np.std(y)<1e-12: return None
    o=np.argsort(x); n=len(x); best=None
    if n<12:
        k=max(3,int(.6*n)); splits=[(o[:k],o[k:]),(o[n-k:],o[:n-k])]
    else: splits=[(o[:int(.35*n)],o[int(.7*n):]),(o[int(.65*n):],o[:int(.3*n)])]
    for tr,te in splits:
        if len(te)<1: continue
        bp=fit_auto(x,y,tr)
        if bp is None: continue
        f,p=bp; pred=f(x,*p)
        res=np.abs(y-pred); q=np.quantile(res[tr],0.9)+max(1e-3*np.std(y),0.01*np.median(np.abs(y[tr])))+1e-12; conf=np.mean(res[te]/q)
        pos=(np.abs(y)>0)&(np.abs(pred)>1e-12); lr=np.abs(np.log(np.clip(np.abs(y)/np.clip(np.abs(pred),1e-12,None),1e-12,None)))
        L=np.full(n,np.nan); L[pos]=lr[pos]; bd=np.nanmedian(L[te])/(max(np.nanmedian(L[tr]),0.01)+1e-6)
        ir=1-np.sum((y[tr]-pred[tr])**2)/(np.sum((y[tr]-np.mean(y[tr]))**2)+1e-12)
        if best is None or ir>best[0]: best=(ir,conf,bd)
    if best is None: return None
    _,conf,bd=best
    return 0.5*(min(conf/3.0,3)+min(bd/4.0,3))
cal_idx=rng.choice(len(Xc),400,replace=False)
CAL_FULL=[];CAL_SMALL=[]
for i in cal_idx:
    c=Xc[i]; yn=c[:,1]*(1+0.01*rng.standard_normal(len(c)))   # 1% measurement noise: calibration reflects MEASURED classical data
    s=stat_score(c[:,0],yn)
    if s is not None: CAL_FULL.append(s)
    sel=rng.choice(len(c),8,replace=False); s2=stat_score(c[sel,0],yn[sel])
    if s2 is not None: CAL_SMALL.append(s2)
CAL_FULL=np.array(CAL_FULL);CAL_SMALL=np.array(CAL_SMALL)
def pctl(v,ref): return float((ref<v).mean())
def score(x,y):
    n=len(x); s=stat_score(x,y)
    return None if s is None else pctl(s,CAL_SMALL if n<20 else CAL_FULL)
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
boyle=np.array(json.load(open(SC+'boyle_1662.json')))
gm_a=np.array([421.8,671.1,1070.4,1882.7]); gm_P=np.array([1.769,3.551,7.155,16.689])
print('=== STATISTICAL component (leakage-free, classical-only size-matched calibration) ===')
for name,x,y,tr in [('FIRAS [break]',nu,I,1),('specific heat Cu [break]',Tc,Cp,1),('Bertozzi [break]',KEb,b2,1),
 ('Onnes [break]',To,Ro,1),('Kepler control',ak,Pk,0),('Boyle 1662 control (25 real pts)',boyle[:,0],boyle[:,1],0),
 ('Galilean moons control (4 real pts)',gm_a,gm_P,0)]:
    v=score(np.asarray(x,float),np.asarray(y,float))
    print('  %-36s score=%s  %s'%(name,'%.2f'%v if v is not None else 'n/a','FLAG' if (v or 0)>0.5 else 'classical'))

# ---------- suite + blind corpus with the improved statistical engine ----------
from sklearn.metrics import roc_auc_score
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
sc=[];lab=[]
for l,fns in [(0,CLASS),(1,BREAK)]:
    for fn in fns:
        for s in range(3):
            r=np.random.RandomState(s); xx=np.sort(r.uniform(0.1,6,200)); yy=fn(xx)*(1+0.02*r.randn(200))
            v=score(xx,yy)
            if v is not None: sc.append(v); lab.append(l)
print('\nsuite (n=%d) statistical AUROC = %.3f'%(len(lab),roc_auc_score(np.array(lab),np.array(sc))))
import re as _re
exec(open(SC+'build_classical_corpus.py',encoding='utf-8').read().split('if __name__')[0])
sys.path.insert(0,'D:/Physics Fundation model/sr_model/data'); import gen_dataside as GD
MASTER='D:/Physics Fundation model/dataset_20260531/_extract_master/master_20260616/master_nodes.jsonl'
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
    Xv=np.asarray(pc['X'],float); yv=np.asarray(pc['y'],float)
    if Xv.ndim!=2 or Xv.shape[0]<40: continue
    cors=[abs(np.corrcoef(Xv[:,i],yv)[0,1]) if np.std(Xv[:,i])>0 else 0 for i in range(Xv.shape[1])]
    i=int(np.argmax(cors))
    if cors[i]<0.2: continue
    neg.append((Xv[:,i],yv*(1+0.01*rng2.standard_normal(len(yv)))))
POS=[(BREAK[0],(0.05,15)),(BREAK[0],(0.05,9)),(BREAK[1],(0.02,0.995)),(BREAK[2],(0.02,0.995)),
 (lambda b:1/np.sqrt(np.clip(1-b**2,1e-4,1)),(0.02,0.99)),(BREAK[3],(0.08,20)),(BREAK[4],(0.0,4.0)),
 (BREAK[5],(0.2,3.0)),(BREAK[0],(0.02,20)),(BREAK[6],(2.0,15))]
sc2=[];lab2=[]
for fn,(a,b) in POS:
    for s in range(3):
        r=np.random.RandomState(s); x=np.sort(r.uniform(a,b,220)); y=fn(x)*(1+0.03*r.randn(220))
        v=score(x,y)
        if v is not None: sc2.append(v); lab2.append(1)
for x,y in neg:
    v=score(x,y)
    if v is not None: sc2.append(v); lab2.append(0)
sc2=np.array(sc2);lab2=np.array(lab2)
tp=((sc2>0.5)&(lab2==1)).sum(); fn_=((sc2<=0.5)&(lab2==1)).sum(); fp=((sc2>0.5)&(lab2==0)).sum(); tn=((sc2<=0.5)&(lab2==0)).sum()
print('blind corpus statistical AUROC = %.3f  | recall=%.2f FPR=%.2f (TP=%d FN=%d FP=%d TN=%d)'%(
 roc_auc_score(lab2,sc2),tp/max(tp+fn_,1),fp/max(fp+tn,1),tp,fn_,fp,tn))
