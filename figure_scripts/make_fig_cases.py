"""Historical measurements under the 1899 horizon, as one panel system.

Panels a-d are the four measured cases; panel e is the Millikan photoelectric
relation, which sits with them because it is the case where a generic shape is
compatible while a supplied incumbent prediction is not.  Supersedes the
separate breakdown and limit figures; the panel code is unchanged apart from
axes placement and panel lettering.
"""
import numpy as np, json, matplotlib
from pathlib import Path
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
 'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'font.size':8.5,'axes.titlesize':9.5,'axes.labelsize':9,
 'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.4,
 'axes.linewidth':0.8,'axes.edgecolor':'#333333','axes.labelcolor':'#1a1a1a','text.color':'#1a1a1a',
 'xtick.color':'#333333','ytick.color':'#333333',
 'xtick.direction':'out','ytick.direction':'out',
 'xtick.major.size':3.0,'ytick.major.size':3.0,'xtick.major.width':0.8,'ytick.major.width':0.8,
 'xtick.minor.size':1.7,'ytick.minor.size':1.7,'xtick.minor.width':0.6,'ytick.minor.width':0.6,
 'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,
 'axes.grid':True,'axes.grid.axis':'y','grid.color':'#d3d3d3','grid.linewidth':0.5,'grid.alpha':0.7,
 'legend.frameon':False,'legend.handlelength':1.5,'legend.handletextpad':0.6,
 'legend.labelspacing':0.35,'legend.borderpad':0.3,'legend.borderaxespad':0.4,
 'lines.solid_capstyle':'round','figure.facecolor':'white','savefig.facecolor':'white',
 'figure.dpi':600,'savefig.dpi':600})
BL='#0072B2'; RD='#D55E00'; PU='#8a6db1'; GY='#7f8c8d'; K='#000000'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'

def plab(A,l,x=-0.135,y=1.13):
    A.text(x,y,l,transform=A.transAxes,fontsize=11,fontweight='bold',va='top',ha='left')
def badge(A,txt,color=RD,xy=(0.985,0.03),ha='right',va='bottom',fs=7.0):
    fc='#fdf1e8' if color==RD else '#eaf1f8'
    A.text(xy[0],xy[1],txt,transform=A.transAxes,ha=ha,va=va,fontsize=fs,color=color,
           bbox=dict(boxstyle='round,pad=0.3',fc=fc,ec=color,lw=0.7))
def leg(A,**kw):
    kw.setdefault('frameon',True); kw.setdefault('facecolor','white')
    kw.setdefault('edgecolor','none'); kw.setdefault('framealpha',0.88)
    return A.legend(**kw)

def load2(fn):
    r=[]
    for line in open(fn):
        s=line.strip()
        if not s or s[0].isalpha() or s.startswith('#'): continue
        p=s.replace(',',' ').split()
        try: r.append((float(p[0]),float(p[1])))
        except: pass
    a=np.array(r); return a[:,0],a[:,1]

fig=plt.figure(figsize=(7.087,8.46))
gs=fig.add_gridspec(3,2,height_ratios=[1.0,1.0,0.80],
                    left=0.085,right=0.985,top=0.955,bottom=0.055,wspace=0.30,hspace=0.46)
A=fig.add_subplot(gs[0,0]); B=fig.add_subplot(gs[0,1])
C=fig.add_subplot(gs[1,0]); D=fig.add_subplot(gs[1,1])
E=fig.add_subplot(gs[2,:])

# ---- a: FIRAS blackbody -------------------------------------------------
nu,I=load2(ROOT/'data'/'firas_monopole.txt'); o=np.argsort(nu); x=nu[o]; y=I[o]; lo=x<4
aRJ=np.sum(y[lo]*x[lo]**2)/np.sum(x[lo]**4)
xx=np.linspace(0,x.max(),200)
A.plot(xx,aRJ*xx**2,color=BL,lw=1.8,label='classical Rayleigh$-$Jeans $\\propto\\nu^{2}$',zorder=2)
A.scatter(x,y,s=11,c=K,zorder=3,label='COBE-FIRAS (published spectrum)')
A.set_ylim(-48,700); A.set_yticks([0,200,400,600]); A.set_xlim(-0.5,22)
A.text(3.2,468,'classical over-predicts\n$\\sim\\!5\\times10^{2}$–$3\\times10^{3}\\times$',
       color=RD,fontsize=7.4,va='top',ha='left',linespacing=1.4)
A.set_title('Blackbody $\\rightarrow$ quantum radiation',pad=6); plab(A,'a')
badge(A,'exploratory 0.95\nno incumbent interior: no boundary',xy=(0.985,0.03),ha='right',va='bottom',fs=6.3)
A.set_xlabel('frequency $\\nu$ (cm$^{-1}$)'); A.set_ylabel('intensity (MJy sr$^{-1}$)')
leg(A,loc='upper right',bbox_to_anchor=(1.0,1.02))
axins=A.inset_axes([0.575,0.40,0.395,0.245])
axins.scatter(x,y,s=2.5,c=K,zorder=3); axins.plot(xx[xx>0],aRJ*xx[xx>0]**2,color=BL,lw=1.0)
axins.set_yscale('log'); axins.set_ylim(1,1e5); axins.set_xlim(0,22)
axins.set_yticks([1e0,1e2,1e4]); axins.set_xticks([0,10,20])
axins.tick_params(labelsize=5.5,length=2,width=0.6,pad=1.5)
for s in axins.spines.values(): s.set_linewidth(0.6)
axins.grid(False); axins.set_title('log scale',fontsize=6.0,pad=1.5)

# ---- b: Bertozzi --------------------------------------------------------
KE=np.array([.5,1,1.5,4.5,15]); b2=np.array([.752,.828,.922,.974,1.0])
kk=np.linspace(.25,15,80); beta2_classical=2*kk/0.511
B.plot(kk,beta2_classical,color=BL,lw=1.8,label='classical $\\frac{1}{2}mv^{2}$ ($\\beta^{2}\\!=\\!2KE/m_ec^{2}$)',zorder=2)
B.scatter(KE,b2,s=22,c=K,zorder=3,label='Bertozzi 1964 (digitized values)')
B.axhline(1,color=GY,ls=(0,(1,2)),lw=1.0,zorder=1)
B.text(6.2,1.09,'light barrier $\\beta^{2}=1$',fontsize=7.4,color=GY)
B.set_ylim(0,3.45); B.set_xlim(-0.4,15.6)
B.text(1.9,2.35,'classical $\\rightarrow\\beta^{2}\\!=\\!59$,\n$v=7.7c$ at 15 MeV',
       color=RD,fontsize=7.4,va='top',ha='left',linespacing=1.4)
B.set_title('Fast electrons $\\rightarrow$ relativity',pad=6); plab(B,'b')
badge(B,'exploratory 0.66\nno incumbent interior: no boundary',fs=6.3)
B.set_xlabel('kinetic energy (MeV)'); B.set_ylabel('$\\beta^{2}=v^{2}/c^{2}$')
leg(B,loc='upper right',bbox_to_anchor=(1.0,1.02))

# ---- c: specific heat ---------------------------------------------------
C.set_xscale('log')
mats=json.load(open(ROOT/'data'/'nist_cp_coef.json'))
def cp(coef,T): L=np.log10(T); return 10**np.clip(sum(cc*L**k for k,cc in enumerate(coef)),-30,30)
C.axvspan(4,40,color=RD,alpha=0.06,lw=0)
_grey_done=False
for i,row in enumerate(mats):
    name,coef,rng=row[0],row[1],row[2]; lo,hi=rng
    T=np.logspace(np.log10(max(lo,4)),np.log10(hi),160); Cp=cp(coef,T); plateau=np.median(Cp[T>0.7*hi])
    r=Cp/plateau
    if name.startswith('Copper'): C.plot(T,r,color=K,lw=1.8,zorder=4,label='OFHC copper (NIST reference curve)')
    else:
        lab='5 other NIST materials (Al, Pt, SS, Be)' if not _grey_done else None
        C.plot(T,r,color=GY,lw=0.9,alpha=0.65,zorder=2,label=lab); _grey_done=True
C.plot([4,300],[1,1],color=BL,lw=1.6,ls=(0,(5,2)),zorder=3,label='classical Dulong$-$Petit (const.)')
C.set_ylim(0,1.42); C.set_yticks([0,0.2,0.4,0.6,0.8,1.0]); C.set_xlim(3.6,340)
C.text(4.4,0.70,'reference curves fall at low $T$\n(Cu over-predicts\n$\\sim\\!3.7\\times10^{3}$ at 4 K)',
       color=RD,fontsize=7.4,va='top',ha='left',linespacing=1.4)
C.set_title('Specific heat $\\rightarrow$ quantum thermodynamics',pad=6); plab(C,'c')
badge(C,'exploratory 0.95\nlocalization candidate',fs=6.3)
C.set_xlabel('temperature $T$ (K)'); C.set_ylabel('$C_p / C_p^{\\,\\mathrm{Dulong-Petit}}$')
leg(C,loc='upper left',bbox_to_anchor=(0.0,1.02))

# ---- d: Onnes 1911 ------------------------------------------------------
T=np.array([4.00,4.10,4.15,4.19,4.21,4.25,4.30,4.35,4.40])
R=np.array([np.nan,np.nan,np.nan,np.nan,0.110,0.118,0.126,0.134,0.142])
norm=T>4.20; Rtc=0.11
D.axvspan(4.0,4.20,color=RD,alpha=0.06,lw=0); D.axvline(4.20,color=RD,ls=(0,(4,2)),lw=0.9,zorder=1)
D.plot([4.0,4.20],[Rtc,Rtc],color=BL,lw=1.7,ls=(0,(5,2)),zorder=2,label='classical residual floor (Drude)')
D.plot([4.20,4.42],[Rtc,0.142],color=BL,lw=1.7,ls=(0,(5,2)),zorder=2)
D.errorbar(T[norm],R[norm],yerr=0.006,fmt='o',ms=3.6,c=K,elinewidth=0.9,capsize=2,
           zorder=3,label='Onnes 1911 Hg (read from figure)')
D.scatter(T[~norm],np.full((~norm).sum(),0.004),marker='v',s=22,c=K,zorder=3)
D.text(4.015,0.011,'$R<10^{-5}\\,\\Omega$ (upper limits)',fontsize=7.0,color=K)
D.set_ylim(-0.010,0.215); D.set_yticks([0,0.04,0.08,0.12,0.16]); D.set_xlim(3.985,4.435)
D.text(4.015,0.082,'$T_c=4.2$ K: classical\npredicts finite resistance;\nreal $R\\to0$ ($>\\!10^{4}\\times$)',
       color=RD,fontsize=7.0,va='top',ha='left',linespacing=1.4)
D.set_title('Superconductivity $\\rightarrow$ resistance collapse',pad=6); plab(D,'d')
badge(D,'exploratory 0.66\nlocalization candidate; censored',fs=6.3)
D.set_xlabel('temperature $T$ (K)'); D.set_ylabel('resistance $R$ ($\\Omega$)')
leg(D,loc='upper left',bbox_to_anchor=(-0.005,1.015))

# ---- e: Millikan photoelectric -----------------------------------------
he=4.124e-15; nu0=4.39e14
nu=np.array([5.49,7.41,8.21,9.59])*1e14
V=he*(nu-nu0)
constant_level=float(np.mean(V))
xx=np.linspace(nu0,10e14,50)
E.axhline(0,color=GY,ls=(0,(1,2)),lw=0.9,zorder=1)
E.plot(xx/1e14,he*(xx-nu0),color=BL,lw=1.7,zorder=2,
       label='$V_{\\rm stop}=(h/e)(\\nu-\\nu_0)$, slope $=+h/e$')
E.hlines(constant_level,4.0,10.0,color=RD,lw=1.6,ls=(0,(4,2)),zorder=2,
         label='declared constant prediction (fitted level)')
E.scatter(nu/1e14,V,s=22,c=K,zorder=4,label='Millikan 1916 Na (digitized values)')
E.plot([nu0/1e14],[0],'v',color=RD,ms=5.5,zorder=5)
E.annotate('extrapolated zero $\\nu_0$',xy=(nu0/1e14,0.02),xytext=(4.62,0.60),color=RD,fontsize=7.4,
           ha='left',va='bottom',
           arrowprops=dict(arrowstyle='->',color=RD,lw=0.9,shrinkA=2,shrinkB=3,relpos=(0.0,0.0)))
E.text(7.15,0.15,'slope recovers $h/e$ to 0.3%; generic shape: compatible.\n'
                 'supplied constant model: incompatible',
       fontsize=7.4,color='#4d4d4d',va='bottom',ha='left',linespacing=1.4)
E.set_xlim(4.0,10.05); E.set_ylim(-0.58,2.72)
E.set_xticks([4,5,6,7,8,9,10]); E.set_yticks([-0.5,0,0.5,1.0,1.5,2.0,2.5])
E.set_title('Generic shape compatible while a supplied prediction fails',pad=6)
plab(E,'e',x=-0.062,y=1.16)
E.set_xlabel('frequency $\\nu$ ($10^{14}$ Hz)'); E.set_ylabel('stopping potential $V_{\\rm stop}$ (V)')
leg(E,loc='upper left',bbox_to_anchor=(0.0,0.995),fontsize=6.8)

OUT.mkdir(exist_ok=True)
for ext in ('png','pdf','svg'):
    fig.savefig(OUT/f'figure_cases.{ext}')
print(OUT/'figure_cases.png')
