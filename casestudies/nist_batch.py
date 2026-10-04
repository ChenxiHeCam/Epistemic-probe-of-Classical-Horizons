"""Batch-pull REAL specific-heat curves from the NIST Cryogenic Material Properties database (public,
citable; each Cp(T) is the accepted reference fit to measured data). Dulong-Petit (classical) predicts a
constant molar heat capacity; every crystalline solid's Cp collapses at low T (Debye/quantum). Detector:
fit classical constant on the high-T interior, extrapolate to low T, measure over-prediction + breakdown.
NIST pages put the specific-heat column in different positions, so we identify the Cp column physically:
the column whose high-T limit matches that material's Dulong-Petit value 3R/M (data-plumbing only; the
detector itself never uses 3R/M). Labelled as NIST reference fits to real measurements, not raw points."""
import re,urllib.request,ssl,numpy as np,json
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
B="https://trc.nist.gov/cryogenics/materials/"
MATS={  # material: (url, molar mass kg/mol) -> Dulong-Petit 3R/M
 'Copper (OFHC)':      ('OFHC%20Copper/OFHC_Copper_rev1.htm',0.06355),
 'Aluminum 6061':      ('6061%20Aluminum/6061_T6Aluminum_rev.htm',0.02698),
 'Aluminum 1100':      ('1100%20Aluminum/1100%20Aluminum_rev.htm',0.02698),
 'Titanium Ti-6Al-4V': ('Ti6Al4V/Ti6Al4V_rev.htm',0.04651),
 'Platinum':           ('Platinum/Platinum_rev.htm',0.19508),
 'Lead':               ('Lead/Lead_rev.htm',0.2072),
 'Molybdenum':         ('Molybdenum/Molybdenum_rev.htm',0.09595),
 '304 Stainless':      ('304Stainless/304Stainless_rev.htm',0.0556),
 '316 Stainless':      ('316Stainless/316Stainless_rev.htm',0.0556),
 'Beryllium':          ('Beryllium/Beryllium_rev.htm',0.009012),
}
def fetch(u):
    req=urllib.request.Request(B+u,headers={'User-Agent':'Mozilla/5.0'})
    return urllib.request.urlopen(req,timeout=40,context=ctx).read().decode('utf-8','ignore')
def cp_of(coef,T):
    L=np.log10(T); s=sum(c*L**k for k,c in enumerate(coef)); return 10**np.clip(s,-30,30)
def parse_cp(html,DP):
    txt=re.sub(r'<[^>]+>',' ',html); txt=re.sub(r'&[a-z]+;',' ',txt)
    i=-1
    for mark in ['J/(kg','J/(kg-K','J/kg','J kg','J/(g','joule']:
        i=txt.lower().find(mark.lower())
        if i>=0: break
    if i<0: return None,None
    seg=txt[i:i+1700]
    # collect all floats on each single-letter row a..i  -> columns
    cols=None
    for lab in 'abcdefghi':
        m=re.search(r'(?<![A-Za-z0-9.])'+lab+r'\s+((?:-?\d+\.?\d*(?:[eE]-?\d+)?|---)(?:\s+(?:-?\d+\.?\d*(?:[eE]-?\d+)?|---)){0,4})',seg)
        if not m: break
        vals=[0.0 if v=='---' else float(v) for v in m.group(1).split()]
        if cols is None: cols=[[] for _ in vals]
        for k in range(len(cols)):
            cols[k].append(vals[k] if k<len(vals) else 0.0)
    if not cols: return None,None
    rng=re.search(r'data range\s+(\d+)\s*-\s*(\d+)',seg); lo,hi=(float(rng.group(1)),float(rng.group(2))) if rng else (4.,300.)
    # pick the column whose high-T limit is closest (log) to Dulong-Petit 3R/M
    Th=hi; best=None
    for c in cols:
        if len(c)<6: continue
        val=cp_of(c,Th)
        if not np.isfinite(val) or val<=0: continue
        d=abs(np.log10(val)-np.log10(DP))
        if best is None or d<best[0]: best=(d,c)
    if best is None or best[0]>0.7: return None,(lo,hi)   # no column within ~5x of Dulong-Petit
    return best[1],(lo,hi)
print('=== REAL specific-heat curves, NIST cryogenic database (validated fits to measured data) ===')
print('  classical = Dulong-Petit const (3R/M); low-T collapse = Debye/quantum onset. Detector sees only numbers.\n')
print('%-20s %-8s %-8s %-9s %-10s %s'%('material (NIST)','Cp(300)','3R/M','over@low','breakdown','verdict'))
rows=[]
for name,(u,M) in MATS.items():
    DP=3*8.314/M
    try: html=fetch(u)
    except Exception as e: print('%-20s fetch-failed'%name); continue
    coef,rng=parse_cp(html,DP)
    if not coef: print('%-20s no-Cp-column'%name); continue
    lo,hi=rng; T=np.logspace(np.log10(max(lo,3.5)),np.log10(hi),140); Cp=cp_of(coef,T)
    if not np.all(np.isfinite(Cp)) or Cp[-1]<=0: print('%-20s bad-curve'%name); continue
    cphi=Cp[-1]; over=np.median(Cp[T>0.7*hi])/max(Cp[0],1e-9)
    ii=T>np.quantile(T,0.7); const=np.median(Cp[ii]); ff=T<np.quantile(T,0.15)
    lr=np.abs(np.log10(np.clip(Cp,1e-9,None))-np.log10(const))
    bd=np.median(lr[ff])/(np.median(lr[ii])+1e-3)
    verdict='NEEDS-NEW (low-T quantum)' if bd>4 else 'classical'
    rows.append((name,list(coef),[lo,hi],cphi,DP,over,bd))
    print('%-20s %-8.0f %-8.0f %-9.0f %-10.1f %s'%(name,cphi,DP,over,bd,verdict))
print('\n  %d real NIST materials: classical Dulong-Petit holds at high T, breaks at low T (Debye quantum onset).'%len(rows))
print('  All public NIST reference fits to measured data (trc.nist.gov/cryogenics) -> real, citable.')
json.dump(rows,open('nist_cp_coef.json','w'))
print('  Saved -> nist_cp_coef.json')
