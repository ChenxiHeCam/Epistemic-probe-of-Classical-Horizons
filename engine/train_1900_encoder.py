"""Step 2: train a strictly 1900-PURE data encoder by contrastive learning on classical point clouds ONLY.
Two random subsamples of the same classical relation = a positive pair; different relations = negatives
(InfoNCE). The encoder never sees a post-1900 functional form, so its latent z_d is a genuine 1900-oracle
representation. GPU memory capped at 70%."""
import sys,os,numpy as np,torch,torch.nn as nn
import legacy_paths  # repository-relative paths and external inputs
legacy_paths.add_repo_root()
from models.encoders import DataEncoder
SC=legacy_paths.SC
if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(0.70); DEV='cuda'
else: DEV='cpu'
MAXV=16
D=np.load(SC+'classical_pointclouds.npz',allow_pickle=True); X=D['X']   # (M,128,2)
M,NP,_=X.shape; print('loaded %d classical point clouds (%d pts each) on %s'%(M,NP,DEV))
X=torch.tensor(X,dtype=torch.float32)
enc=DataEncoder(max_vars=MAXV,d=256,n_isab=6,dim_len=0,n_tokens=16,log_feats=True,class_feats=True,robust_norm=True).to(DEV)
opt=torch.optim.Adam(enc.parameters(),lr=3e-4,weight_decay=1e-5)
def views(batch_idx,k=96):
    # two subsamples of the same relation, each with an INDEPENDENT random x- and y-rescaling, so the
    # encoder must learn a SCALE-INVARIANT representation of functional form (not the data's range/units).
    B=len(batch_idx); pts=torch.zeros(2*B,k,MAXV+1)
    for j,i in enumerate(batch_idx):
        cloud=X[i]
        for v in range(2):
            sel=torch.randperm(NP)[:k]; sub=cloud[sel].clone()
            sx=float(np.exp(np.random.uniform(-1.5,1.5))); sy=float(np.exp(np.random.uniform(-1.5,1.5)))
            pts[v*B+j,:,0]=sub[:,0]*sx; pts[v*B+j,:,MAXV]=sub[:,1]*sy
    vm=torch.zeros(2*B,MAXV); vm[:,0]=1; pm=torch.ones(2*B,k)
    return pts.to(DEV),vm.to(DEV),pm.to(DEV)
def infonce(z,tau=0.1):
    z=nn.functional.normalize(z,dim=-1); B=z.shape[0]//2
    sim=z@z.t()/tau; sim.fill_diagonal_(-1e9)
    tgt=torch.arange(2*B,device=z.device); tgt=(tgt+B)%(2*B)     # positive = the paired view
    return nn.functional.cross_entropy(sim,tgt)
enc.train(); BS=256; STEPS=2500
for step in range(STEPS):
    idx=np.random.choice(M,BS,replace=False)
    pts,vm,pm=views(idx)
    z=enc(pts,vm,pm,dims=None)
    loss=infonce(z); opt.zero_grad(); loss.backward(); opt.step()
    if step%100==0: print('  step %d  infonce loss %.3f'%(step,loss.item()))
torch.save({'state':enc.state_dict(),'d':256,'max_vars':MAXV},legacy_paths.out('encoder_1900.pt'))
print('saved 1900-pure encoder -> encoder_1900.pt')
