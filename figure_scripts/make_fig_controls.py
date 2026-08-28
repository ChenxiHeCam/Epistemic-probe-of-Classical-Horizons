"""Figure 3 - three REAL classical controls + detector benchmark + leakage-free learned component.
(a) Solar-System Kepler, (b) Galilean moons of Jupiter, (c) Boyle 1662 pressure-volume (25 real points):
    all three fitted by the classical form and scored well below threshold (0.02 / 0.00 / 0.38).
(d) Benchmark against standard detectors (n=36, bootstrap CIs); (e) leakage-free learned component
    vs the statistical detector on the held-out suite (n=180).
Layout pass: 12-column GridSpec - a/b/c on the top row, d (7 cols) and e (5 cols) filling the bottom row,
so there is no empty cell. Typography identical to Figs 2, 4, 5."""
import os
import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------- shared house style (identical in make_fig_breakdowns/controls/rigor/limit) -------------
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
 'figure.dpi':300,'savefig.dpi':300})
BL='#0072B2'; RD='#D55E00'; PU='#8a6db1'; GY='#7f8c8d'; K='#000000'
def plab(A,l,x=-0.135,y=1.13):
    A.text(x,y,l,transform=A.transAxes,fontsize=11,fontweight='bold',va='top',ha='left')
def badge(A,txt,color=BL,xy=(0.985,0.03),ha='right',va='bottom',fs=7.0):
    fc='#fdf1e8' if color==RD else '#eaf1f8'
    A.text(xy[0],xy[1],txt,transform=A.transAxes,ha=ha,va=va,fontsize=fs,color=color,
           bbox=dict(boxstyle='round,pad=0.3',fc=fc,ec=color,lw=0.7))
def leg(A,**kw):
    kw.setdefault('frameon',True); kw.setdefault('facecolor','white')
    kw.setdefault('edgecolor','none'); kw.setdefault('framealpha',0.88)
    return A.legend(**kw)
# ---------------------------------------------------------------------------------------------------------
SC=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data','')

fig=plt.figure(figsize=(7.2,5.2))
gs=fig.add_gridspec(2,12,wspace=2.9,hspace=0.52,left=0.075,right=0.985,top=0.925,bottom=0.085)
axA=fig.add_subplot(gs[0,0:4]); axB=fig.add_subplot(gs[0,4:8]); axC=fig.add_subplot(gs[0,8:12])
axD=fig.add_subplot(gs[1,0:7]); axE=fig.add_subplot(gs[1,7:12])

# ---- a: Solar-System Kepler ----
nm=['Me','V','E','Ma','J','S','U','N']
a=np.array([0.387,0.723,1.0,1.524,5.203,9.537,19.19,30.07]); P=np.array([0.241,0.615,1.0,1.881,11.86,29.46,84.01,164.8])
sl,ic=np.polyfit(np.log10(a),np.log10(P),1)
A=axA; A.set_xscale('log'); A.set_yscale('log')
aa=np.logspace(np.log10(0.3),np.log10(40),50)
A.plot(aa,10**ic*aa**sl,color=BL,lw=1.7,zorder=2,label='classical fit, $P\\propto a^{%.3f}$'%sl)
A.scatter(a,P,s=17,c=K,zorder=3,label='Solar System (measured)')
for n,x0,y0 in zip(nm,a,P): A.annotate(n,(x0,y0),textcoords='offset points',xytext=(4,-7.5),fontsize=6.6,color=GY)
A.set_xlim(0.28,45); A.set_ylim(0.13,520)
A.set_title('Solar-System orbits',pad=6); plab(A,'a')
A.set_xlabel('semi-major axis $a$ (AU)'); A.set_ylabel('period $P$ (yr)')
leg(A,loc='upper left',bbox_to_anchor=(-0.005,1.015)); badge(A,'engine score 0.02')

# ---- b: Galilean moons ----
mn=['Io','Eu','Ga','Ca']; ga=np.array([421.8,671.1,1070.4,1882.7]); gP=np.array([1.769,3.551,7.155,16.689])
s2,i2=np.polyfit(np.log10(ga),np.log10(gP),1)
B=axB; B.set_xscale('log'); B.set_yscale('log')
gg=np.logspace(np.log10(380),np.log10(2100),50)
B.plot(gg,10**i2*gg**s2,color=BL,lw=1.7,zorder=2,label='classical fit, $P\\propto a^{%.3f}$'%s2)
B.scatter(ga,gP,s=17,c=K,zorder=3,label='Galilean moons (measured)')
for n,x0,y0 in zip(mn,ga,gP): B.annotate(n,(x0,y0),textcoords='offset points',xytext=(4,-7.5),fontsize=6.6,color=GY)
B.set_xlim(370,2500); B.set_ylim(1.1,34)
B.set_title('Galilean moons of Jupiter',pad=6); plab(B,'b')
B.set_xlabel('semi-major axis ($10^{3}$ km)'); B.set_ylabel('period (days)')
leg(B,loc='upper left',bbox_to_anchor=(-0.005,1.015)); badge(B,'engine score 0.00')

# ---- c: Boyle 1662 ----
bo=np.array(json.load(open(SC+'boyle_1662.json'))); bv=bo[:,0]; bp=bo[:,1]
s3,i3=np.polyfit(np.log10(bv),np.log10(bp),1)
C=axC
vv=np.linspace(11,50,80)
C.plot(vv,10**i3*vv**s3,color=BL,lw=1.7,zorder=2,label='classical fit, $P\\propto V^{%.3f}$'%s3)
C.scatter(bv,bp,s=14,c=K,zorder=3,label='Boyle 1662 (25 real points)')
C.set_xlim(9,51); C.set_ylim(20,152); C.set_yticks([20,40,60,80,100,120,140])
C.set_title("Boyle's 1662 pressure$-$volume data",pad=6); plab(C,'c')
C.set_xlabel('volume (arbitrary units)'); C.set_ylabel('pressure (inches Hg)')
leg(C,loc='upper right',bbox_to_anchor=(1.0,1.015)); badge(C,'engine score 0.38',xy=(0.985,0.45))

# ---- d: benchmark against standard detectors ----
D=axD
names=['GP\ndiscrep.','CUSUM','RESET','breakdown','conformal','statistical\ndetector']
vals=np.array([0.58,0.67,0.73,0.72,0.77,0.80]); ci=np.array([0.11,0.12,0.13,0.13,0.14,0.14])
xp=np.arange(len(names))
D.bar(xp,vals,color=[GY]*5+[BL],edgecolor='none',width=0.62,zorder=2)
D.errorbar(xp,vals,yerr=ci,fmt='none',ecolor='#2b2b2b',elinewidth=0.9,capsize=2.5,zorder=3)
for i,v in enumerate(vals): D.text(i,v+ci[i]+0.014,'%.2f'%v,ha='center',fontsize=7.4)
D.axhline(0.5,color='#666666',lw=0.8,ls=(0,(1,2)),zorder=1)
D.set_ylim(0.45,1.03); D.set_yticks([0.5,0.6,0.7,0.8,0.9,1.0]); D.set_xlim(-0.62,5.62)
D.set_xticks(xp); D.set_xticklabels(names,fontsize=7.2)
D.tick_params(axis='x',length=0)
D.set_ylabel('AUROC (fair suite, $n=36$)')
D.set_title('Benchmark against standard detectors',pad=6); plab(D,'d',x=-0.078)

# ---- e: leakage-free learned component ----
E=axE
E.bar([0,1],[0.62,0.83],color=[GY,PU],edgecolor='none',width=0.5,zorder=2)
for i,v in enumerate([0.62,0.83]): E.text(i,v+0.012,'%.2f'%v,ha='center',fontsize=7.4)
E.axhline(0.5,color='#666666',lw=0.8,ls=(0,(1,2)),zorder=1)
E.set_xlim(-0.7,1.7); E.set_ylim(0.45,0.95); E.set_yticks([0.5,0.6,0.7,0.8,0.9])
E.set_xticks([0,1]); E.set_xticklabels(['statistical','learned\n(1900-pure)'],fontsize=7.6)
E.tick_params(axis='x',length=0)
E.set_ylabel('AUROC (held-out, $n=180$)')
E.set_title('Leakage-free learned component',pad=6); plab(E,'e',x=-0.11)

plt.savefig('figure_controls.png',bbox_inches='tight')
print('saved figure_controls.png ; Kepler %.4f  Galilean %.4f  Boyle %.4f'%(sl,s2,s3))
