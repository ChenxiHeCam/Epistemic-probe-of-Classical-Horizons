"""E2, leakage-free: the learned engine using the strictly 1900-PURE encoder (trained by contrastive
learning on classical point clouds only, never seeing a post-1900 form). z_d(full) + linear probe ->
breakdown vs classical. Because the encoder is period-pure, this is a genuine 1900-oracle learned engine."""
import sys,os,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from models.encoders import DataEncoder
SC=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data','')
DEV='cpu'; MAXV=16; DIML=6
ck=torch.load(SC+'encoder_1900.pt',map_location=DEV)
enc=DataEncoder(max_vars=MAXV,d=256,n_isab=6,dim_len=ck.get('dim_len',DIML),n_tokens=16,log_feats=True,class_feats=True,robust_norm=True).to(DEV)
enc.load_state_dict(ck['state']); enc.eval()
@torch.no_grad()
def zd(x,y,xd=None,yd=None):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)&(np.abs(y)<1e18); x,y=x[m],y[m]
    if len(x)<20: return None
    pts=np.zeros((1,len(x),MAXV+1),np.float32); pts[0,:,0]=x; pts[0,:,MAXV]=y
    dims=torch.zeros(1,MAXV+1,DIML)
    if xd is not None: dims[0,0]=torch.tensor(xd,dtype=torch.float32)
    if yd is not None: dims[0,MAXV]=torch.tensor(yd,dtype=torch.float32)
    z=enc(torch.tensor(pts),torch.tensor([[1.]+[0.]*(MAXV-1)]),torch.ones(1,len(x)),dims=dims)
    z=z[0].numpy(); return z if np.all(np.isfinite(z)) else None
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
Xf=[];Y=[]
for lab,fns,rg in [(0,CLASS,(0.1,6)),(1,BREAK,(0.1,6))]:
    for fn in fns:
        for s in range(3):
            r=np.random.RandomState(s); a,b=rg; xx=np.sort(r.uniform(a,b,200)); yy=fn(xx)*(1+0.02*r.randn(200))
            z=zd(xx,yy)
            if z is not None: Xf.append(z);Y.append(lab)
Xf=np.array(Xf);Y=np.array(Y)
clf=LogisticRegression(max_iter=3000,C=1.0)
p=cross_val_predict(clf,Xf,Y,cv=5,method='predict_proba')[:,1]
print('=== E2 (1900-PURE encoder): z_d + linear probe, 5-fold CV ===')
print('  encoder trained contrastively on CLASSICAL point clouds only (no post-1900 form seen)')
print('  n=%d (%d classical, %d breakdown)  cross-validated AUROC = %.3f'%(len(Y),(Y==0).sum(),(Y==1).sum(),roc_auc_score(Y,p)))
clf.fit(Xf,Y)
FREQ=[0,0,-1,0,0,0]; INTEN=[1,0,-2,0,0,0]; TEMP=[0,0,0,0,1,0]; CP=[0,2,-2,0,-1,0]; RES=[1,2,-3,-2,0,0]; LEN=[0,1,0,0,0,0]; TIME=[0,0,1,0,0,0]; Z6=[0]*6
REAL={'FIRAS blackbody':(lambda v:v**3/(np.exp(clp(v))-1+1e-9),(0.5,11),FREQ,INTEN),
 'relativity KE':(lambda b:1/np.sqrt(np.clip(1-b**2,1e-4,1))-1,(0.75,0.99),Z6,Z6),
 'specific heat Debye':(lambda T:(1/np.clip(T,1e-2,None))**2*np.exp(np.clip(1/np.clip(T,1e-2,None),0,40))/(np.exp(np.clip(1/np.clip(T,1e-2,None),0,40))-1)**2,(0.1,15),TEMP,CP),
 'superconductor step':(lambda T:np.where(T>1,0.1+0.05*(T-1),1e-6),(0.2,3),TEMP,RES),'Kepler [classical]':(lambda a:a**1.5,(0.3,30),LEN,TIME)}
print('  real-case transfer (with real units):')
for name,(fn,(a,b),xd,yd) in REAL.items():
    r=np.random.RandomState(0); xx=np.sort(r.uniform(a,b,200)); yy=fn(xx)*(1+0.02*r.randn(200)); z=zd(xx,yy)
    pr=clf.predict_proba(z[None])[0,1] if z is not None else float('nan')
    print('    %-24s P(breakdown)=%.2f  %s'%(name,pr,'FLAG' if pr>0.5 else 'classical'))
