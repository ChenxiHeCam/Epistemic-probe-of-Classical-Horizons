"""REAL-DATA test: COBE-FIRAS CMB monopole spectrum (Fixsen et al. 1996), the most precise blackbody ever
measured. Apply the symbolic-extrapolation detector: fit a classical form on the LOW-frequency
(Rayleigh-Jeans) regime, extrapolate to high frequency, compare to data. Classical (RJ ~ nu^2) diverges
while the data peaks and falls -> high frontier residual = 'needs new physics'. This is the 1900 oracle on real data."""
import numpy as np
from scipy.optimize import curve_fit
import warnings; warnings.filterwarnings('ignore')
rows=[]
for line in open('firas_monopole.txt'):
    line=line.strip()
    if not line or line.startswith('#'): continue
    p=line.split()
    if len(p)>=2:
        try: rows.append((float(p[0]),float(p[1]),float(p[3])))
        except: pass
D=np.array(rows); nu=D[:,0]; I=D[:,1]; unc=D[:,2]   # cm^-1, MJy/sr, uncertainty kJy/sr
print('REAL FIRAS CMB spectrum: %d points, freq %.2f-%.2f cm^-1, peak intensity %.1f MJy/sr'%(len(nu),nu.min(),nu.max(),I.max()))
# ---- symbolic-extrapolation detector on the real curve ----
FORMS={'lin':(lambda x,a,b:a*x+b,[1,0]),'quad':(lambda x,a,b,c:a*x**2+b*x+c,[1,0,0]),
 'cubic':(lambda x,a,b,c,d:a*x**3+b*x**2+c*x+d,[1,0,0,0]),'power':(lambda x,a,b,c:a*np.abs(x)**b+c,[1,2,0]),
 'exp':(lambda x,a,b,c:a*np.exp(np.clip(b*x,-60,60))+c,[1,-.5,0]),'rec':(lambda x,a,b,c:a/(x+b)+c,[1,1,0])}
o=np.argsort(nu); x=nu[o]; y=I[o]; n=len(x)
tr=np.arange(0,int(.45*n)); te=np.arange(int(.55*n),n)   # fit RJ regime (low freq), extrapolate to peak+Wien
ys=np.std(y)
best=1e9; bname=None; bpred=None
for nm,(f,p0) in FORMS.items():
    try:
        popt,_=curve_fit(f,x[tr],y[tr],p0=p0,maxfev=6000); ri=np.sqrt(np.mean((f(x[tr],*popt)-y[tr])**2))
        if ri<best: best=ri; bname=nm; bpred=f(x,*popt); bpopt=popt
    except: pass
frontier_resid=np.sqrt(np.mean((bpred[te]-y[te])**2))/ys
print('\nbest classical form fit on low-freq interior: "%s" (interior RMSE=%.2f MJy/sr)'%(bname,best))
print('NOVELTY = normalized frontier residual = %.3f  (>>0 => classical extrapolation breaks => NEEDS NEW PHYSICS)'%frontier_resid)
# ---- explicit Rayleigh-Jeans (classical) fit + extrapolation = the UV catastrophe, quantified ----
RJ=lambda x,a: a*x**2
aRJ,_=curve_fit(RJ,x[tr],y[tr],p0=[1]);
rj_pred=RJ(x,*aRJ)
print('\n--- Rayleigh-Jeans (classical, I~nu^2) fit to low-freq, extrapolated ---')
print(' freq(cm^-1)   data(MJy/sr)   RJ-classical-pred   ratio pred/data')
for i in range(0,n,max(1,n//12)):
    print('   %6.2f       %8.1f        %10.1f        %6.1fx'%(x[i],y[i],rj_pred[i],rj_pred[i]/max(y[i],1e-6)))
imax=n-1
print('\nAt the highest measured frequency (%.1f cm^-1): classical RJ predicts %.0f MJy/sr, data is %.1f MJy/sr'%(x[imax],rj_pred[imax],y[imax]))
print('=> classical over-predicts by %.0fx (the ultraviolet catastrophe), detected from REAL data.'%(rj_pred[imax]/max(y[imax],1e-6)))
