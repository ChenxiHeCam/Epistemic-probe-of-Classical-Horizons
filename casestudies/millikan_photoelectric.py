"""REAL photoelectric data — Millikan 1916 (Phys. Rev. 7, 355), public domain. Balance (stopping)
potentials for four clean sodium lines, read from the paper's Table I (p.372) by extrapolating each
photocurrent-vs-potential set to zero deflection (the balance point Millikan located graphically in
Fig. 5). Frequencies from the mercury line wavelengths. Two readings:
 (A) as a pure DATA SHAPE, (nu, V) is a straight line — a classically-representable form — so a
     shape-only detector correctly does NOT flag it (a real instance of the shape-degeneracy limit).
 (B) against the CLASSICAL PHYSICAL PREDICTION that electron energy is set by intensity, i.e.
     independent of frequency (slope 0), the real data depart: slope = h/e != 0, with a threshold.
This is the historically-correct classical expectation, and the departure recovers the quantum onset."""
import numpy as np
h_e=6.626e-34/1.602e-19   # h/e = 4.136e-15 V s (reference)
c=2.998e8
# --- REAL data from Millikan Table I: line wavelength (Angstrom) -> balance potential V (volts) ---
# balance V = potential at zero photocurrent deflection (linear extrapolation of the table's last points)
LINES={5461:2.08, 4047:1.35, 3650:0.92, 3126:0.32}   # four clean, monotonic sodium lines
nu=np.array([c/(A*1e-10) for A in LINES])             # Hz
V =np.array(list(LINES.values()))
print('=== REAL photoelectric data: Millikan 1916, sodium (Phys. Rev. 7, 355), 4 lines ===')
print('  line(A)   frequency(1e14 Hz)   balance potential V (volts)')
for A,(f,v) in zip(LINES,zip(nu,V)): print('   %-7d  %-18.3f  %.2f'%(A,f/1e14,v))
# slope of the real V-nu line = h/e
s14,intcpt=np.polyfit(nu/1e14,V,1); slope=s14/1e14   # fit on scaled freq to avoid ill-conditioning
fitV=slope*nu+intcpt; R2=1-np.sum((V-fitV)**2)/np.sum((V-V.mean())**2)
print('\n  (A) DATA SHAPE: least-squares (nu,V) is a straight line, R^2=%.4f'%R2)
print('      |slope| = %.3e V.s   vs   h/e = %.3e V.s   (agree to %.1f%%)'%(abs(slope),h_e,100*abs(abs(slope)-h_e)/h_e))
print('      -> a pure shape detector fits this line and does NOT flag new physics (shape-degeneracy limit).')
print('\n  (B) vs CLASSICAL PREDICTION (electron energy independent of frequency => slope 0, V=const):')
Vconst=V.mean()
# breakdown = systematic, frequency-growing residual of the constant (classical) hypothesis
res=V-Vconst; conf=np.sqrt(np.mean(res**2))/ (0.02)   # 0.02 V = Millikan's stated per-point error
print('      classical constant V=%.2f leaves residuals %s'%(Vconst,np.round(res,2)))
print('      residual is monotone in nu (corr=%.3f) and %.0fx the 0.02 V measurement error -> classical FAILS.'%(
      np.corrcoef(nu,res)[0,1], conf))
print('      the real slope h/e is exactly the frequency-dependence classical physics forbids (Einstein 1905 / quantum).')
# threshold (Millikan's contact-PD-corrected sodium value, from Fig. 6)
print('\n  Threshold nu_0 = 43.9e13 Hz (Millikan Fig. 6, sodium; contact-PD corrected): below it, NO emission')
print('  -> classical predicts emission at all frequencies given enough intensity; the real cutoff localizes the onset.')
print('\n  VERDICT: NEEDS-NEW *given the classical prediction* (energy ⊥ frequency); ABSTAIN on shape alone.')
print('  Role: a REAL historical example of the shape-degeneracy limit — the quantum content')
print('  is in the frequency-dependence/threshold, not the (linear) data shape. Complements FIRAS/specific-heat')
print('  (where the SHAPE itself departs) and the tunnelling toy (where it does not).')
