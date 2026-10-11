import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from res import *
import matplotlib; matplotlib.use('Agg')
FIG=ROOT/'article'/'v13'/'figures'; FIG.mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt, matplotlib.patches as mp
plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'grid.linestyle':'--','figure.dpi':200,'savefig.dpi':300})
DARK='#2C2C2A'
BLUE,AMB,GRN,RED,GRY,PUR='#185FA5','#BA7517','#0F6E56','#9C2B22','#666666','#6A3FA0'
W20=W   # contient w=4,8,12,20
# ---------- Fig 1 : schéma des deux conceptions ----------
fig,ax=plt.subplots(figsize=(9.2,3.9)); ax.set_xlim(0,39); ax.set_ylim(0,4.6); ax.axis('off')
ax.add_patch(mp.Rectangle((0,0.3),12,3.7,fc='#E6F1FB',ec='none')); ax.text(6,4.12,'observed weeks 0–11\n(features)',ha='center',va='bottom',fontsize=9,color=BLUE)
ax.add_patch(mp.Rectangle((12,0.3),27,3.7,fc='#F4F4F2',ec='none')); ax.text(25.5,4.12,'future weeks (not observed at prediction time)',ha='center',va='bottom',fontsize=9,color=GRY)
ax.axvline(12,color=DARK,lw=2); ax.text(12,0.05,'end of window  $w$ = 12',ha='center',va='top',fontsize=9.5,fontweight='bold')
rows=[(3.2,5,'a'),(2.1,18,'b'),(1.0,39,'c')]
for y0,last,k in rows:
    xs=np.arange(0,last); ax.scatter(xs+.5,[y0]*len(xs),s=22,color=BLUE,zorder=3)
ax.text(-0.4,3.2,'a',ha='right',va='center',fontweight='bold'); ax.text(-0.4,2.1,'b',ha='right',va='center',fontweight='bold'); ax.text(-0.4,1.0,'c',ha='right',va='center',fontweight='bold')
ax.scatter([5.5],[3.2],marker='X',s=110,color=RED,zorder=4); ax.text(12.6,3.2,'unregisters (week 5)  →  Design A: positive, already recorded; Design B: excluded',va='center',fontsize=8.6,color=RED)   # à droite de la ligne (ne la croise plus)
ax.scatter([18.5],[2.1],marker='X',s=110,color=RED,zorder=4); ax.text(19.6,2.1,'withdraws (week 18)  →  positive in both designs',va='center',fontsize=8.6,color=RED)
ax.text(19.6,0.62,'completes the course  →  negative in both designs',va='center',fontsize=8.6,color=GRN)   # au-dessus des points (chevauchement corrigé)
ax.set_title('Design A (all students) vs. Design B (leakage-aware window)',fontsize=11,fontweight='bold',loc='left')
plt.tight_layout(); plt.savefig(str(FIG/'fig1_designs.png'),bbox_inches='tight'); plt.close()
# ---------- Fig 2 : AUC vs fenêtre, conception A vs B ----------
ws=[4,8,12,20,39]; fig,ax=plt.subplots(figsize=(7.6,4.8))
for name,c,mk in [('Aggregated(8)',GRY,'o'),('Raw sequence',BLUE,'s'),('Last active week (1 var.)',AMB,'^')]:
    ax.errorbar(ws,[m(A[str(w)][name]['auc']) for w in ws],yerr=[s(A[str(w)][name]['auc']) for w in ws],color=c,marker=mk,lw=2,capsize=2,elinewidth=.9,label=f'Design A — {name}')
wb=[4,8,12,20]
for key,lab,c,mk in [('Aggregated(8) [B1-like]','Aggregated(8)',GRY,'o'),('C1 raw sequence flattened','Raw sequence',BLUE,'s'),('C3 PCA-64','PCA-64',PUR,'D')]:
    xs=[w for w in wb if key in W20[str(w)]]; ax.errorbar(xs,[m(W20[str(w)][key]['auc']) for w in xs],yerr=[s(W20[str(w)][key]['auc']) for w in xs],color=c,marker=mk,lw=1.6,ls='--',mfc='white',capsize=2,elinewidth=.9,label=f'Design B — {lab}')
# embedding et encodeur non entraîné : moyenne de cinq graines d'autoencodeur (results_seedavg.json), comme dans les tableaux 4 et 5
sa=lambda w,k: SA[str(w)]['models'][k]['fold_auc']
ax.errorbar(wb,[m(sa(w,'C4 untrained (5 seeds)')) for w in wb],yerr=[s(sa(w,'C4 untrained (5 seeds)')) for w in wb],color=GRN,marker='v',lw=1.6,ls='--',mfc='white',capsize=2,elinewidth=.9,label='Design B — Untrained LSTM encoder (5 seeds)')
ax.errorbar(wb,[m(sa(w,'Embedding (5 seeds)')) for w in wb],yerr=[s(sa(w,'Embedding (5 seeds)')) for w in wb],color=RED,marker='o',lw=2.2,ls='--',capsize=2,elinewidth=.9,label='Design B — Trained autoencoder (5 seeds)')
ax.set_xticks(ws); ax.set_xlabel('Observation window (weeks)'); ax.set_ylabel('AUC-ROC'); ax.set_ylim(0.5,0.96); ax.axhline(.5,color='k',lw=.8)
ax.set_title('Same models, two evaluation designs',fontweight='bold'); ax.legend(fontsize=7.6,ncol=1,loc='center left',bbox_to_anchor=(1.01,0.5),frameon=False)
plt.tight_layout(); plt.savefig(str(FIG/'fig2_auc_vs_window.png'),bbox_inches='tight'); plt.close()
# ---------- Fig 3 : efficacité en étiquettes ----------
fr=['0.01','0.02','0.05','0.1','0.25','1.0']; xv=[float(f)*100 for f in fr]; fig,ax=plt.subplots(figsize=(7.2,4.6))
series=[('AE + LogReg',[L[f]['AE + LogReg']['auc'] for f in fr],RED,'o','-'),
        ('PCA-64 + LogReg',[X[f]['PCA-64 + LogReg'] for f in fr],PUR,'D','--'),
        ('Untrained LSTM encoder + LogReg',[X[f]['Untrained LSTM enc. + LogReg'] for f in fr],GRN,'v','--'),
        ('Raw sequence + LogReg (C = 0.01)',[X[f]['Raw sequence + LogReg (C=0.01)'] for f in fr],AMB,'s','--'),
        ('Raw sequence + LogReg (C = 1)',[X[f]['Raw sequence + LogReg (C=1)'] for f in fr],AMB,'x',':'),
        ('Raw sequence + XGBoost',[L[f]['Raw sequence + XGBoost']['auc'] for f in fr],BLUE,'s','-'),
        ('Aggregated(8) + XGBoost',[L[f]['Aggregated(8) + XGBoost']['auc'] for f in fr],GRY,'o','-')]
for lab,v,c,mk,ls in series: ax.errorbar(xv,[m(a) for a in v],yerr=[s(a) for a in v],color=c,marker=mk,ls=ls,lw=1.8,capsize=2,elinewidth=.7,alpha=.9,label=lab)
b3=[(float(f)*100,L[f]['B3 supervised LSTM']['auc']) for f in fr if 'B3 supervised LSTM' in L[f]]
ax.errorbar([a for a,_ in b3],[m(b) for _,b in b3],yerr=[s(b) for _,b in b3],color='k',marker='*',ms=10,ls='none',capsize=2,elinewidth=.7,label='B3 supervised LSTM (evaluated at 4 shares)')   # marqueurs seuls : B3 n'est pas évalué à 2 % ni à 10 %
ax.set_xscale('log'); ax.set_xticks(xv); ax.set_xticklabels([f'{x:g}%' for x in xv]); ax.set_xlabel('Share of training labels used'); ax.set_ylabel('AUC-ROC (12-week window)')
ax.set_title('Label efficiency, week-12 window',fontweight='bold'); ax.legend(fontsize=7.8,loc='center left',bbox_to_anchor=(1.01,.5),frameon=False)
plt.tight_layout(); plt.savefig(str(FIG/'fig3_label_efficiency.png'),bbox_inches='tight'); plt.close()
# ---------- Fig 4 : présentations retenues ----------
mods=['AE + LogReg','AE + XGBoost','Raw sequence + XGBoost','Aggregated(8) + XGBoost','B3 supervised LSTM']; cols=[RED,'#D98A87',BLUE,GRY,'k']
pres=sorted(P.keys()); fig,ax=plt.subplots(figsize=(8.2,4.4)); bw=.16
for i,(mo,c) in enumerate(zip(mods,cols)):
    vals=[P[p][mo]['auc'] for p in pres]+[np.mean([P[p][mo]['auc'] for p in pres])]
    ax.bar(np.arange(len(vals))+(i-2)*bw,vals,bw,color=c,label=mo)
ax.set_xticks(range(len(pres)+1)); ax.set_xticklabels([f'held-out {p}' for p in pres]+['mean']); ax.set_ylim(.5,.68); ax.axhline(.5,color='k',lw=.8)
ax.set_ylabel('AUC-ROC (12-week window; axis starts at 0.5)'); ax.set_title('Leave-one-presentation-out',fontweight='bold'); ax.legend(fontsize=7.8,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.12))
plt.tight_layout(); plt.savefig(str(FIG/'fig4_lopo.png'),bbox_inches='tight'); plt.close()
print('figures ok')
