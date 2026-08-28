"""Figure 1 — method schematic. A single, domain-agnostic pipeline: notation-free numbers ->
fit the pre-1900 classical dictionary on the interior -> extrapolate -> score the frontier departure
-> three verdicts. The same pipeline is applied unchanged to every phenomenon (radiation, mechanics,
thermodynamics, condensed matter)."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams
rcParams.update({'font.size':9.5,'figure.dpi':200})
BL='#0072B2'; RD='#D55E00'; PU='#8a6db1'; GY='#7f8c8d'; GD='#b7950b'; GN='#009E73'
LB='#eaf1f8'; LP='#f3eef8'; LG='#fdf6e3'
fig,ax=plt.subplots(figsize=(13,4.6)); ax.set_xlim(0,13); ax.set_ylim(-0.35,4.6); ax.axis('off')

def box(x,y,w,h,text,fc=LB,ec=BL,fs=9,head=None,ha_y=None):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.12',
                                fc=fc,ec=ec,lw=1.4,zorder=2))
    if head is None:
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,zorder=3)
    else:
        ax.text(x+w/2,y+h-0.22,head,ha='center',va='center',fontsize=fs+0.5,
                fontweight='bold',color=ec,zorder=3)
        ax.text(x+w/2,y+h*0.36,text,ha='center',va='center',fontsize=fs,zorder=3)

def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,
                                 lw=1.3,color=GY,zorder=1,shrinkA=0,shrinkB=0))

# --- stage 1: input point cloud (thumbnail sits fully inside the box, above the label) ---
box(0.20,1.35,2.15,1.75,'notation-free\nnumeric data\n(a point cloud)',fc='#f5f5f5',ec=GY)
axm=ax.inset_axes([0.030,0.590,0.110,0.115])
rs=np.random.RandomState(0); xx=np.linspace(0,1,30)
axm.scatter(xx,xx**2+0.02*rs.randn(30),s=2.2,c='k'); axm.axis('off')
axm.set_facecolor('none')

# --- stage 2: interior identification ---
box(2.80,1.35,2.05,1.75,'identify the interior\n(pre-registered;\nregime where a\nclassical law holds)')

# --- stage 3: the statistical detector delivers verdicts; learned component is independent evidence ---
box(5.30,2.45,2.55,1.15,'fit pre-1900 form, extrapolate;\nconformal $\\oplus$ residual ratio;\ncalibrated on real classical data',
    fc=LB,ec=BL,fs=8,head='Statistical detector')
box(5.30,0.55,2.55,1.15,'set-transformer latent (51k classical\nformulas) + deformation probe;\nno post-1900 law in training',
    fc=LP,ec=PU,fs=8,head='Learned (1900-pure)')

# --- stage 4: verdicts (from the statistical detector) ---
box(9.05,3.00,2.30,0.80,'classical everywhere',fc='#e6f4ee',ec=GN,fs=9)
box(9.05,1.82,2.30,0.80,'localized breakdown\n+ boundary',fc='#fdeee4',ec=RD,fs=9)
box(9.05,0.64,2.30,0.80,'no classical regime',fc='#fef6e6',ec=GD,fs=9)

arrow(2.42,2.22,2.72,2.22)
arrow(4.92,2.22,5.22,3.02); arrow(4.92,2.22,5.22,1.12)
arrow(7.92,3.02,8.97,3.40); arrow(7.92,3.02,8.97,2.22); arrow(7.92,3.02,8.97,1.04)
ax.text(6.575,0.30,'independent leakage-free evidence',fontsize=8,color=PU,va='center',ha='center',style='italic')

ax.text(6.5,4.20,'One domain-agnostic pipeline, applied unchanged to every phenomenon',
        ha='center',va='center',fontsize=10.5,color='k')
ax.text(6.5,-0.10,'radiation · mechanics · thermodynamics · condensed matter — same detector, only the numbers change',
        ha='center',va='center',fontsize=8.5,color=GY,style='italic')
plt.savefig('figure_schematic.png',dpi=200,bbox_inches='tight'); print('saved figure_schematic.png')
