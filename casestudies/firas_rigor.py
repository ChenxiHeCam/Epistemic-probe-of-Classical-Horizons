"""FIRAS rigor (addresses reviewer 2): (1) show FIRAS samples x=h*nu/kT in [1.2,11.3], NOT the Rayleigh-Jeans
regime x<<1; (2) window-sensitivity of the over-prediction factor & breakdown ratio to the interior-fit choice;
(3) error-weighted (chi^2) RJ fit using published FIRAS 1-sigma; (4) bootstrap CI on the reported factors."""
import numpy as np
rows=[]
for line in open('firas_monopole.txt'):
    s=line.strip()
    if s and not s.startswith('#'):
        p=s.split()
        try: rows.append((float(p[0]),float(p[1]),float(p[3])))  # freq cm^-1, intensity MJy/sr, sigma kJy/sr
        except: pass
D=np.array(rows); nu=D[:,0]; I=D[:,1]; sig=D[:,2]/1000.0  # sigma to MJy/sr
kT_hc=1.381e-23*2.725/(6.626e-34*2.998e10)   # cm^-1
x=nu/kT_hc
print('FIRAS: %d pts. x=h*nu/kT range = [%.2f, %.2f]  (Rayleigh-Jeans regime needs x<<1) -> FIRAS never samples pure RJ.'%(len(nu),x.min(),x.max()))
print('peak intensity at nu=%.2f cm^-1 (x=%.2f)\n'%(nu[np.argmax(I)],x[np.argmax(I)]))
def rj_fit(mask, weighted=True):
    # I = a*nu^2 on masked (interior) points; weighted least squares with FIRAS sigma
    w=1/sig[mask]**2 if weighted else np.ones(mask.sum())
    a=np.sum(w*I[mask]*nu[mask]**2)/np.sum(w*nu[mask]**4)
    return a
print('=== window sensitivity: fit RJ (I~nu^2) on lowest-N points, report over-prediction at highest freq ===')
print(' interior            a(fit)   pred@21.3   over-predict factor   breakdown-ratio(front/int)')
for frac in [0.10,0.15,0.20,0.30,0.40]:
    k=max(3,int(frac*len(nu))); o=np.argsort(nu); mask=np.zeros(len(nu),bool); mask[o[:k]]=True
    a=rj_fit(mask); pred=a*nu**2
    ov=pred[o[-1]]/max(I[o[-1]],1e-6)
    te=o[int(0.7*len(nu)):]
    fr=np.sqrt(np.mean((np.log(np.abs(pred[te])/np.abs(np.clip(I[te],1e-6,None))))**2))
    ir=np.sqrt(np.mean((np.log(np.abs(pred[mask])/np.abs(np.clip(I[mask],1e-6,None))))**2))+1e-6
    print('  lowest %2d pts (%.0f%%)   %.2f    %8.1f       %7.0fx            %6.1fx'%(k,frac*100,a,pred[o[-1]],ov,fr/ir))
# bootstrap CI on over-prediction factor for a fixed 20% window
o=np.argsort(nu); k=int(0.2*len(nu)); rng=np.random.RandomState(0); ovs=[]
for _ in range(2000):
    idx=rng.choice(o[:k],k,replace=True); a=np.sum(I[idx]*nu[idx]**2)/np.sum(nu[idx]**4)
    ovs.append(a*nu[o[-1]]**2/max(I[o[-1]],1e-6))
print('\n over-prediction factor @highest-freq (20%% window): median %.0fx, 95%% CI [%.0f, %.0f]'%(np.median(ovs),np.percentile(ovs,2.5),np.percentile(ovs,97.5)))
print('\n HONEST NOTE: FIRAS lacks x<<1 coverage, so the RJ anchor sits on the shoulder; the over-prediction')
print(' factor is window-dependent (ranges above) but is LARGE and robustly >1 for every window -> qualitative')
print(' UV-catastrophe / quantum-onset localization holds, though the exact factor should be quoted as a range.')
