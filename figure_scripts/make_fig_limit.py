"""Figure 5 - the intrinsic limit, shown by Millikan's photoelectric line.
Stopping potential INCREASES with frequency, V_stop=(h/e)(nu-nu0), positive slope +h/e, low-frequency
threshold nu0. The (nu,V) data are a classically-representable straight line; a shape-only detector fits it
and does not flag. The quantum content is the nonzero slope h/e and the threshold, visible only against the
classical prediction that electron energy is set by intensity (flat in nu).
Typography identical to Figs 2, 3, 4."""
import numpy as np, matplotlib
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
def leg(A,**kw):
    kw.setdefault('frameon',True); kw.setdefault('facecolor','white')
    kw.setdefault('edgecolor','none'); kw.setdefault('framealpha',0.88)
    return A.legend(**kw)
# ---------------------------------------------------------------------------------------------------------

he=4.124e-15; nu0=4.39e14                         # Millikan sodium: h/e, threshold
nu=np.array([5.49,7.41,8.21,9.59])*1e14           # four clean Na lines (Hg wavelengths)
V=he*(nu-nu0)                                      # stopping potentials (positive slope)

fig,A=plt.subplots(figsize=(3.6,2.9))
xx=np.linspace(nu0,10e14,50)
A.axhline(0,color=GY,ls=(0,(1,2)),lw=0.9,zorder=1)
A.plot(xx/1e14,he*(xx-nu0),color=BL,lw=1.7,zorder=2,
       label='$V_{\\rm stop}=(h/e)(\\nu-\\nu_0)$, slope $=+h/e$')
A.hlines(0.0,4.0,nu0/1e14,color=RD,lw=1.6,ls=(0,(4,2)),zorder=2)
A.scatter(nu/1e14,V,s=22,c=K,zorder=4,label='Millikan 1916 Na (real)')
A.plot([nu0/1e14],[0],'v',color=RD,ms=5.5,zorder=5)
A.annotate('threshold $\\nu_0$',xy=(nu0/1e14,0.02),xytext=(4.62,0.60),color=RD,fontsize=7.4,
           ha='left',va='bottom',
           arrowprops=dict(arrowstyle='->',color=RD,lw=0.9,shrinkA=2,shrinkB=3,relpos=(0.0,0.0)))
A.text(4.02,-0.40,'classical prediction: electron energy set\nby intensity, independent of frequency',
       color=RD,fontsize=7.4,va='bottom',ha='left',linespacing=1.4)
A.text(6.75,0.30,'slope recovers $h/e$ to 0.3%;\nshape alone: classical.\nQuantum $=$ nonzero slope\n$+$ threshold',
       fontsize=7.4,color='#4d4d4d',va='bottom',ha='left',linespacing=1.4)
A.set_xlim(4.0,10.05); A.set_ylim(-0.58,2.72)
A.set_xticks([4,5,6,7,8,9,10]); A.set_yticks([-0.5,0,0.5,1.0,1.5,2.0,2.5])
A.set_title('Intrinsic limit: a classical-looking line\nthat is genuinely quantum',pad=6)
A.set_xlabel('frequency $\\nu$ ($10^{14}$ Hz)'); A.set_ylabel('stopping potential $V_{\\rm stop}$ (V)')
leg(A,loc='upper left',bbox_to_anchor=(-0.005,1.015))

plt.tight_layout(pad=0.5)
plt.savefig('figure_limit.png',bbox_inches='tight')
print('saved figure_limit.png ; V_stop=',np.round(V,3),' slope/he check=%.3f'%(np.polyfit(nu,V,1)[0]/he))
