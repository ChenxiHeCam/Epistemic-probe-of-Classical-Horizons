"""Full-corpus 1900-pure encoder (v2): larger encoder (d=512), more data, longer training, and a
MULTI-TASK objective = contrastive (scale-augmented views of the same relation) + auxiliary prediction of
the formula's functional-form signature (exp/log/trig/sqrt/pow/div/n_terms/hyperbolic) from z_d. The
signature supervision teaches z_d to encode functional form richly (the ingredient that makes the large
pre-trained model strong), while all data is pre-1900 classical -> strictly leakage-free. GPU <=70%."""
import os
import sys,numpy as np,torch,torch.nn as nn
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.encoders import DataEncoder
SC=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data','')
if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(0.70); DEV='cuda'
else: DEV='cpu'
MAXV=16; Dd=256; DIML=6
D=np.load(SC+'classical_pointclouds.npz',allow_pickle=True)
X=torch.tensor(D['X'],dtype=torch.float32); S=torch.tensor(D['sigs'],dtype=torch.float32)
DXY=torch.tensor(D['dimsxy'],dtype=torch.float32)   # (M,2,6): [dim(x), dim(y)] -> UNITS carried into the encoder
M,NP,_=X.shape; print('loaded %d classical clouds, sig %d, dims %s, on %s'%(M,S.shape[1],tuple(DXY.shape),DEV))
enc=DataEncoder(max_vars=MAXV,d=Dd,n_isab=6,dim_len=DIML,n_tokens=16,log_feats=True,class_feats=True,robust_norm=True).to(DEV)
aux=nn.Linear(Dd,S.shape[1]).to(DEV)
opt=torch.optim.Adam(list(enc.parameters())+list(aux.parameters()),lr=3e-4,weight_decay=1e-5)
def views(idx,k=64):
    B=len(idx); pts=torch.zeros(2*B,k,MAXV+1); dims=torch.zeros(2*B,MAXV+1,DIML)
    for j,i in enumerate(idx):
        cloud=X[i]
        for v in range(2):
            sel=torch.randperm(NP)[:k]; sub=cloud[sel].clone()
            sx=float(np.exp(np.random.uniform(-1.5,1.5))); sy=float(np.exp(np.random.uniform(-1.5,1.5)))
            pts[v*B+j,:,0]=sub[:,0]*sx; pts[v*B+j,:,MAXV]=sub[:,1]*sy
            dims[v*B+j,0]=DXY[i,0]; dims[v*B+j,MAXV]=DXY[i,1]   # units unchanged by rescaling
    vm=torch.zeros(2*B,MAXV); vm[:,0]=1; pm=torch.ones(2*B,k)
    return pts.to(DEV),vm.to(DEV),pm.to(DEV),dims.to(DEV)
QSIZE=8192
queue=nn.functional.normalize(torch.randn(QSIZE,Dd,device=DEV),dim=-1)  # memory bank of negatives (MoCo-style)
def infonce_mb(z,tau=0.1):
    global queue
    z=nn.functional.normalize(z,dim=-1); B=z.shape[0]//2
    l_batch=z@z.t()/tau; l_batch.fill_diagonal_(-1e9)                    # in-batch negatives
    l_q=z@queue.t().detach()/tau                                        # + thousands of queue negatives
    logits=torch.cat([l_batch,l_q],1); tgt=(torch.arange(2*B,device=z.device)+B)%(2*B)
    loss=nn.functional.cross_entropy(logits,tgt)
    queue=torch.cat([z.detach(),queue],0)[:QSIZE]                       # enqueue current, drop oldest
    return loss
enc.train(); aux.train(); BS=256; STEPS=10000; lam=0.5
for step in range(STEPS):
    idx=np.random.choice(M,BS,replace=False)
    pts,vm,pm,dims=views(idx); z=enc(pts,vm,pm,dims=dims)
    lc=infonce_mb(z)
    sig_t=S[idx].repeat(2,1).to(DEV); ls=nn.functional.binary_cross_entropy_with_logits(aux(z),sig_t)
    loss=lc+lam*ls; opt.zero_grad(); loss.backward(); opt.step()
    if step%300==0: print('  step %d  contrastive %.3f  sig %.3f'%(step,lc.item(),ls.item()))
torch.save({'state':enc.state_dict(),'d':Dd,'max_vars':MAXV,'dim_len':DIML},SC+'encoder_1900.pt')
print('saved full-corpus 1900-pure encoder (d=512, dim_len=6, multi-task) -> encoder_1900.pt')
