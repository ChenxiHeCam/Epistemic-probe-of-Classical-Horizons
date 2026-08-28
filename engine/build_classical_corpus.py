"""Stage 1 of scaling the IMPLICIT dimensional channel on OUR real training corpus.
Source: a knowledge base of historical physics formulas (5.62M nodes; not redistributed),
the SAME KG master the manifold/PISR trained on. Filter to PRE-1900 CLASSICAL domains (user-confirmed
whitelist), parse each variable's si_unit into a dimension vector [M,L,T,I,Theta,mol], and save each
formula's set of variable dimension-vectors. This is the era-pure classical basis for the dimensional
channel (NOT the modern _haiku_v2 side corpus)."""
import os
import json,re,collections,sys
KEEP={'thermodynamics','fluid dynamics','electromagnetism','optics','statistical mechanics',
 'statmech_advanced','mechanics','classical mechanics','astrophysics','acoustics_advanced','acoustics',
 'continuum mechanics','celestial mechanics','gravitation','hydrodynamics','wave physics'}
DROP_KW=re.compile(r'quantum|relativ|hbar|planck|photon|dirac|schrod|spin|boson|fermion|qft|gauge|'
 r'compton|broglie|tunnel|entangl|neutrino|nuclear|radioact|supercond|semiconduct|laser|bose_einstein',re.I)
# ---- SI unit -> dimension vector [M,L,T,I,Theta,mol] ----
BASE={'kg':(1,0,0,0,0,0),'g':(1,0,0,0,0,0),'m':(0,1,0,0,0,0),'s':(0,0,1,0,0,0),'A':(0,0,0,1,0,0),
 'K':(0,0,0,0,1,0),'mol':(0,0,0,0,0,1),'rad':(0,0,0,0,0,0),'sr':(0,0,0,0,0,0),'cd':(0,0,0,0,0,0)}
DERIVED={'N':(1,1,-2,0,0,0),'J':(1,2,-2,0,0,0),'W':(1,2,-3,0,0,0),'Pa':(1,-1,-2,0,0,0),
 'C':(0,0,1,1,0,0),'V':(1,2,-3,-1,0,0),'F':(-1,-2,4,2,0,0),'Ohm':(1,2,-3,-2,0,0),'ohm':(1,2,-3,-2,0,0),
 'S':(-1,-2,3,2,0,0),'T':(1,0,-2,-1,0,0),'Wb':(1,2,-2,-1,0,0),'H':(1,2,-2,-2,0,0),'Hz':(0,0,-1,0,0,0),
 'Bq':(0,0,-1,0,0,0),'eV':(1,2,-2,0,0,0),'bar':(1,-1,-2,0,0,0),'L':(0,3,0,0,0,0),'l':(0,3,0,0,0,0),
 'N_m':(1,2,-2,0,0,0),'Gy':(0,2,-2,0,0,0),'Sv':(0,2,-2,0,0,0),'lm':(0,0,0,0,0,0),'lx':(0,-2,0,0,0,0),
 'P':(1,-1,-1,0,0,0),'St':(0,2,-1,0,0,0),'D':(0,0,1,1,0,0),'deg':(0,0,0,0,0,0),'percent':(0,0,0,0,0,0)}
PREFIX={'Y','Z','E','P','G','M','k','h','da','d','c','m','u','µ','μ','n','p','f','a','z','y'}
UNITS={**BASE,**DERIVED}
def unit_atom(tok):
    tok=tok.strip()
    if not tok or tok in ('1','dimensionless','unitless','none','a.u.','arb','ratio','fraction','#'):
        return (0,0,0,0,0,0)
    m=re.match(r'^([A-Za-zµμΩ_]+)\^?(-?\d+)?$',tok)
    if not m: return None
    u,e=m.group(1),m.group(2)
    e=int(e) if e else 1
    u=u.replace('Ω','Ohm')
    if u in UNITS: dv=UNITS[u]
    elif len(u)>1 and u[0] in PREFIX and u[1:] in UNITS: dv=UNITS[u[1:]]   # strip prefix
    elif len(u)>2 and u[:2]=='da' and u[2:] in UNITS: dv=UNITS[u[2:]]
    else: return None
    return tuple(x*e for x in dv)
def parse_unit(s):
    if s is None: return None
    s=str(s).strip()
    if s in ('','1','dimensionless','unitless','none','-','a.u.','ratio','fraction'): return (0,0,0,0,0,0)
    s=s.replace('·','*').replace('⋅','*').replace('×','*').replace(' per ','/')
    # split numerator/denominator on first '/'
    if '/' in s:
        num,den=s.split('/',1)
    else:
        num,den=s,''
    def factors(part):
        part=part.replace('(','').replace(')','')
        part=re.sub(r'([A-Za-z])(\d)',r'\1^\2',part)   # m2 -> m^2
        return re.split(r'[\*\s]+',part.strip())
    acc=[0,0,0,0,0,0]
    for f in factors(num):
        if not f: continue
        a=unit_atom(f)
        if a is None: return None
        acc=[x+y for x,y in zip(acc,a)]
    for f in factors(den):
        if not f: continue
        a=unit_atom(f)
        if a is None: return None
        acc=[x-y for x,y in zip(acc,a)]
    return tuple(acc)
if __name__=='__main__' and '--test' in sys.argv:
    for u in ['N','J','kg/mol','rad/s','m/s^2','W/(m2 K)','J/(kg K)','V/m','1/s','dimensionless','Pa','T','Hz','m2','kg*m/s^2','mol/L']:
        print('%-14s -> %s'%(u,parse_unit(u)))
    sys.exit()
# ---- stream master, filter classical, parse units ----
F='data/master_nodes.jsonl'  # source knowledge base (not redistributed; see README)
out=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data',''),'w')
kept=0; seen=0; parse_fail=0; dropped_dom=0; dropped_kw=0
uniq=set()
for ln in open(F,encoding='utf-8'):
    seen+=1
    try: r=json.loads(ln)
    except: continue
    dom=(r.get('domain') or '').strip()
    if dom not in KEEP: dropped_dom+=1; continue
    expr=r.get('expr','') or ''
    if DROP_KW.search(expr) or DROP_KW.search(dom): dropped_kw+=1; continue
    vs=r.get('variables') or []
    dims=[]
    ok=True
    for v in vs:
        if not isinstance(v,dict): continue
        d=parse_unit(v.get('si_unit'))
        if d is None: ok=False; break
        dims.append(d)
    if not ok: parse_fail+=1; continue
    if len(dims)<2: continue
    key=tuple(sorted(dims))
    if key in uniq: continue          # dedup identical dimensional signatures
    uniq.add(key)
    out.write(json.dumps({'id':r.get('id'),'dom':dom,'dims':dims})+'\n'); kept+=1
out.close()
print('scanned',seen,'| classical-domain kept-after-parse',kept,'(unique dim-signatures)')
print('  dropped: non-classical-domain',dropped_dom,'| kw-modern',dropped_kw,'| unit-parse-fail',parse_fail)
