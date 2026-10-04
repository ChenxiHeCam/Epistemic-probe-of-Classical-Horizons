"""LEGACY point-cloud generator from a heuristic 'classical-domain' filter.

The upstream filter does not establish publication date or provenance, so the
output is not 1900-pure.  Retained only for reproduction of the old checkpoint.
Use an audited allow-list snapshot from ``build_pre1900_corpus_strict.py`` for a
new period-restricted encoder.
"""
import sys,os,json,warnings; warnings.filterwarnings('ignore')
import legacy_paths  # repository-relative paths and external inputs
legacy_paths.add_gen_dataside()
import numpy as np, gen_dataside as GD
SC=legacy_paths.SC
exec(open(legacy_paths.CORPUS_BUILDER,encoding='utf-8').read().split('if __name__')[0])  # KEEP, DROP_KW
import re
MASTER=legacy_paths.master_graph()
N_TARGET=300000; NP=128
rng=np.random.default_rng(3)
def sig(expr):
    """cheap functional-form signature of a classical formula (auxiliary supervision target)."""
    e=expr.lower()
    return [1.0 if 'exp' in e else 0.0, 1.0 if 'log' in e else 0.0,
            1.0 if re.search(r'sin|cos|tan',e) else 0.0, 1.0 if 'sqrt' in e else 0.0,
            1.0 if '**' in e or '^' in e else 0.0, 1.0 if '/' in e else 0.0,
            float(min(e.count('+')+e.count('-'),6))/6.0, 1.0 if re.search(r'sinh|cosh|tanh',e) else 0.0]
Xs=[]; ids=[]; sigs=[]; dimsxy=[]   # dimsxy: per cloud [dim(x), dim(y)] each a 6-vector [M,L,T,I,Theta,mol]
seen=0
for ln in open(MASTER,encoding='utf-8'):
    if len(Xs)>=N_TARGET: break
    try: r=json.loads(ln)
    except: continue
    if (r.get('domain') or '').strip() not in KEEP: continue
    if DROP_KW.search(r.get('expr','') or ''): continue
    vs=r.get('variables') or []
    if not (2<=len(vs)<=4): continue
    seen+=1
    try: pc=GD.make_one(r['expr'],rng,hint=vs[0].get('sym'))
    except: pc=None
    if not pc: continue
    X=np.asarray(pc['X'],float); y=np.asarray(pc['y'],float)
    if X.ndim!=2 or X.shape[0]<NP: continue
    cors=[abs(np.corrcoef(X[:,i],y)[0,1]) if np.std(X[:,i])>0 else 0 for i in range(X.shape[1])]
    i=int(np.argmax(cors))
    if cors[i]<0.2: continue
    x=X[:,i]; m=np.isfinite(x)&np.isfinite(y)&(np.abs(y)<1e18); x,y=x[m],y[m]
    if len(x)<NP: continue
    sel=rng.choice(len(x),NP,replace=False); x,y=x[sel],y[sel]
    # dimensions of the chosen input variable (x) and the output variable (y), parsed from si_unit
    unit_of={}
    for v in vs:
        if isinstance(v,dict):
            du=parse_unit(v.get('si_unit')); unit_of[v.get('sym')]=list(du) if du is not None else [0]*6
    vn=pc.get('var_names',[]); tgt=pc.get('target','')
    xsym=vn[i] if i<len(vn) else None
    xd=unit_of.get(xsym,[0]*6); yd=unit_of.get(tgt,[0]*6)
    Xs.append(np.stack([x,y],1).astype(np.float32)); ids.append(r.get('id','')); sigs.append(sig(r.get('expr','') or ''))
    dimsxy.append([xd,yd])
Xs=np.array(Xs,dtype=np.float32)   # (M, NP, 2)  each = one classical 1-D relation
np.savez_compressed(legacy_paths.out('classical_pointclouds.npz'), X=Xs, ids=np.array(ids),
                    sigs=np.array(sigs,dtype=np.float32), dimsxy=np.array(dimsxy,dtype=np.float32))
print('generated %d classical point clouds (scanned %d), X %s dims %s -> classical_pointclouds.npz'%(len(Xs),seen,Xs.shape,np.array(dimsxy).shape))
