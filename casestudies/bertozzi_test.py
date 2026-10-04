"""REAL relativity data: Bertozzi 1964 (Am. J. Phys. 32, 551) — kinetic energy vs measured speed of
electrons. Values digitized from the paper / standard textbook reproductions:
  KE (MeV):  0.5,   1.0,   1.5,   4.5,   15
  beta^2=v^2/c^2 (measured): 0.752, 0.828, 0.922, 0.974, 1.0
Classical mechanics: KE = 1/2 m v^2  => v^2 (hence beta^2) is LINEAR in KE through the origin.
Fit that classical law on the low-energy points, extrapolate: it predicts beta^2 growing without bound
(>1, i.e. faster than light) while the data saturate at c -> localized breakdown = special relativity."""
import numpy as np
KE=np.array([0.5,1.0,1.5,4.5,15.0])        # MeV
b2=np.array([0.752,0.828,0.922,0.974,1.0]) # v^2/c^2, measured
# classical: beta^2 = a*KE (from 1/2 m v^2 = KE). Fit slope on the two lowest-energy points.
a=np.sum(b2[:2]*KE[:2])/np.sum(KE[:2]**2)
pred=a*KE
print('=== REAL relativity data — Bertozzi 1964 (KE vs measured v^2/c^2) ===')
print(' classical law: 1/2 m v^2 = KE  =>  beta^2 = a*KE (linear).  Fit a=%.3f on low-energy points.\n'%a)
print('  KE(MeV)   measured beta^2   classical beta^2 (=a*KE)   status')
for i in range(len(KE)):
    st='classical OK' if pred[i]<1.05 and abs(pred[i]/b2[i]-1)<0.2 else ('BREAKS (classical>1=faster than light!)' if pred[i]>1.05 else 'breaks')
    print('   %5.1f       %7.3f            %8.2f              %s'%(KE[i],b2[i],pred[i],st))
ov=pred[-1]/b2[-1]
print('\n at 15 MeV: classical predicts beta^2 = %.1f (v = %.1fc, impossible); data saturates at beta^2 = %.2f'%(pred[-1],np.sqrt(pred[-1]),b2[-1]))
print(' => classical OVER-PREDICTS by %.0fx and violates the light barrier -> new physics (special relativity),'%ov)
print('    localized from REAL data. beta^2 physically cannot exceed 1 -> a limiting speed is detected.')
# breakdown ratio (log-residual frontier/interior)
lr=np.abs(np.log(np.clip(b2,1e-6,None)/np.clip(pred,1e-6,None)))
print('\n breakdown-ratio (frontier/interior log-residual) = %.1f  -> flagged NEEDS-NEW.'%((np.median(lr[-2:]))/(np.median(lr[:2])+1e-6)))
print('\n Source: W. Bertozzi, Am. J. Phys. 32, 551 (1964).')
