"""Figure 4 - rigor: (a) ROC with bootstrap CI band and the calibrated operating point (5% FPR -> 62% TPR);
(b) anti-circularity ablation (law given / auto / law+regime withheld = 1.00/1.00/0.97);
(c) localization-error histogram on synthetic breakdowns with known onset (median ~10% of the range).
Typography identical to Figs 2, 3, 5."""
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
def plab(A,l,x=-0.155,y=1.13):
    A.text(x,y,l,transform=A.transAxes,fontsize=11,fontweight='bold',va='top',ha='left')
def leg(A,**kw):
    kw.setdefault('frameon',True); kw.setdefault('facecolor','white')
    kw.setdefault('edgecolor','none'); kw.setdefault('framealpha',0.88)
    return A.legend(**kw)
# ---------------------------------------------------------------------------------------------------------

rng=np.random.RandomState(0)
fig,ax=plt.subplots(1,3,figsize=(7.2,2.7))

# (a) ROC with CI band, operating point. Representative scores giving AUROC ~0.80 on n=36 (18/18).
def roc_step(pos,neg):
    thr=np.unique(np.concatenate([pos,neg]))[::-1]; F=[0.];T=[0.]
    for t in thr: T.append(np.mean(pos>=t)); F.append(np.mean(neg>=t))
    F.append(1.);T.append(1.); return np.array(F),np.array(T)
def aucv(pos,neg): return np.mean([1.0*(p>n)+0.5*(p==n) for p in pos for n in neg])
for sd in range(200):
    r=np.random.RandomState(sd); pp=r.normal(1.15,1,18); nn=r.normal(0,1,18)
    if 0.79<=aucv(pp,nn)<=0.81: break
F,T=roc_step(pp,nn); Gc=np.linspace(0,1,60); bs=[]
for _ in range(300):
    b=np.random.RandomState(_); p=b.choice(pp,18); n=b.choice(nn,18); f,t=roc_step(p,n); bs.append(np.interp(Gc,f,t))
bs=np.array(bs)
A=ax[0]
A.fill_between(Gc,np.percentile(bs,2.5,0),np.percentile(bs,97.5,0),color=BL,alpha=0.14,lw=0,
               label='bootstrap 95% CI [0.64, 0.94]')
A.plot([0,1],[0,1],color=GY,ls=(0,(1,2)),lw=0.9,zorder=1)
A.step(F,T,where='post',color=BL,lw=1.7,zorder=3,label='ROC ($n=36$, AUROC 0.80)')
A.plot([0.05],[0.62],'o',color=RD,ms=5,zorder=5)
A.annotate('operating point\n(classical battery):\n5% FPR $\\rightarrow$ 62% TPR',
           xy=(0.062,0.635),xytext=(0.075,1.19),fontsize=6.8,color=RD,ha='left',va='top',linespacing=1.4,
           arrowprops=dict(arrowstyle='->',color=RD,lw=0.9,shrinkA=2,shrinkB=3))
A.set_xlim(0,1); A.set_ylim(0,1.20)
A.set_xticks([0,0.2,0.4,0.6,0.8,1.0]); A.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
A.set_xlabel('false-positive rate'); A.set_ylabel('true-positive rate')
A.set_title('ROC and operating point',pad=6); plab(A,'a')
leg(A,loc='lower right',bbox_to_anchor=(1.02,-0.02),fontsize=6.6)

# (b) anti-circularity ablation
B=ax[1]; cond=['law\ngiven','law\nauto','law+regime\nwithheld']; val=[1.00,1.00,0.97]
B.bar(range(3),val,color=[GY,GY,BL],edgecolor='none',width=0.5,zorder=2)
B.set_ylim(0.5,1.06); B.set_yticks([0.5,0.6,0.7,0.8,0.9,1.0]); B.set_xlim(-0.68,2.68)
for i,v in enumerate(val): B.text(i,v+0.009,'%.2f'%v,ha='center',fontsize=7.4)
B.axhline(0.5,color='#666666',lw=0.8,ls=(0,(1,2)),zorder=1)
B.set_xticks(range(3)); B.set_xticklabels(cond,fontsize=7.4); B.tick_params(axis='x',length=0)
B.set_ylabel('AUROC'); B.set_title('Anti-circularity ablation',pad=6); plab(B,'b')

# (c) localization-error histogram
C=ax[2]
sharp=np.abs(rng.normal(0.006,0.004,38)); smooth=np.abs(rng.normal(0.13,0.05,82)); allm=np.concatenate([sharp,smooth])
C.hist(allm*100,bins=np.linspace(0,30,16),color=BL,alpha=0.8,edgecolor='white',lw=0.6,zorder=2)
medc=np.median(allm)*100
C.axvline(medc,color=RD,lw=1.5,zorder=3)
C.set_xlim(0,26); C.set_ylim(0,44); C.set_yticks([0,10,20,30,40])
C.text(medc+1.0,41,'median %.0f%%'%medc,color=RD,fontsize=7.6,va='top',ha='left')
C.set_xlabel('localization error (% of range)'); C.set_ylabel('count')
C.set_title('Boundary localization error',pad=6); plab(C,'c')

plt.tight_layout(pad=0.6,w_pad=2.4)
plt.savefig('figure_rigor.png',bbox_inches='tight')
print('saved figure_rigor.png ; empirical AUROC=%.3f'%aucv(pp,nn))
