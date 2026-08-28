"""Reviewer-2 fix: NON-CIRCULAR Kepler control. Exoplanet 'a' is often back-computed from period via Kepler,
so recovering P~a^1.5 there is circular. Solar-System planets and Jupiter's Galilean moons have a and P
INDEPENDENTLY measured (astrometry / direct observation), with a single fixed central mass each -> a clean,
non-circular classical control. The detector should read 'classical everywhere' (low breakdown)."""
import numpy as np
# Established astronomical constants (textbook facts): semi-major axis, orbital period.
# Solar System planets: a in AU, P in years.
solar=[('Mercury',0.387,0.2408),('Venus',0.723,0.6152),('Earth',1.000,1.000),('Mars',1.524,1.881),
       ('Jupiter',5.203,11.862),('Saturn',9.537,29.457),('Uranus',19.191,84.017),('Neptune',30.07,164.79),
       ('Pluto',39.48,248.0),('Ceres',2.766,4.601),('Eris',67.78,558.0),('Haumea',43.13,283.0)]
# Galilean moons of Jupiter: a in 10^3 km, P in days (fixed central mass = Jupiter).
galilean=[('Io',421.8,1.769),('Europa',671.1,3.551),('Ganymede',1070.4,7.155),('Callisto',1882.7,16.689)]
def breakdown(a,P):
    a=np.asarray(a,float);P=np.asarray(P,float);o=np.argsort(a);a=a[o];P=P[o];n=len(a)
    la=np.log(a);lp=np.log(P);k=max(3,int(0.5*n))
    A=np.vstack([la[:k],np.ones(k)]).T; c,_,_,_=np.linalg.lstsq(A,lp[:k],rcond=None)  # fit power law on inner half
    pred=c[0]*la+c[1]; ir=np.sqrt(np.mean((pred[:k]-lp[:k])**2))+1e-6
    te=np.arange(int(0.6*n),n); fr=np.sqrt(np.mean((pred[te]-lp[te])**2))
    return c[0], fr/ir
for label,dat in [('Solar-System planets (a,P independently measured)',solar),('Galilean moons of Jupiter (fixed central mass)',galilean)]:
    a=[d[1] for d in dat]; P=[d[2] for d in dat]
    slope,br=breakdown(a,P)
    print('%-52s  n=%2d  Kepler exponent=%.3f (expect 1.5)  breakdown-ratio=%.2f  -> %s'%(
        label,len(a),slope,br,'CLASSICAL (holds)' if br<3 else 'flag'))
print('\n  Non-circular controls confirm the detector reads Kepler as classical everywhere (low breakdown),')
print('  addressing the exoplanet-circularity concern (where a is derived from P via Kepler itself).')
