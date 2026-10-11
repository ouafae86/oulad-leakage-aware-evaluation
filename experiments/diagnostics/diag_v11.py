"""Diagnostics du pipeline v11 (cours entier) — reproduit les contrôles faits à la main lors de la critique.
Écrit results/diagnostics/diagnostics_v11.json. Idempotent (FORCE=1 pour recalculer). Lecture seule sur interim/ et results/v11_end_of_course/."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','early_windows'))
from common import *
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
OUTJ=str(ROOT/'results'/'diagnostics'/'diagnostics_v11.json'); skip_if_exists(OUTJ,'diagnostics v11')
os.makedirs(os.path.dirname(OUTJ),exist_ok=True)
V11=ROOT/'results'/'v11_end_of_course'
feats=pd.read_parquet(f'{UP}/aggregated_features.parquet'); emb=np.load(f'{UP}/embeddings.npy'); rec=np.load(V11/'recon_mse.npy')
wk=seq.sum(2); last=np.array([np.max(np.where(r>0)[0]) if (r>0).any() else -1 for r in wk]); aw=feats['active_weeks'].values
o={}
o['auc_last_active_week_nocv']=float(roc_auc_score(y,-last)); o['auc_active_weeks_nocv']=float(roc_auc_score(y,-aw))
o['median_last_active_week']={'non_dropout':float(np.median(last[y==0])),'dropout':float(np.median(last[y==1]))}
o['share_withdrawn_unregistered_before_end']=float(((unreg<7*39)&(y==1)).sum()/(y==1).sum())
# validation interne de l'autoencodeur v11 = les 10 % derniers étudiants (validation_split Keras non mélangé)
cut=int(len(y)*0.9); v=cohort.iloc[cut:]
o['v11_validation_set']={'n':int(len(v)),'dropout_rate':float(y[cut:].mean()),'dropout_rate_rest':float(y[:cut].mean()),'recon_mse_mean':float(rec[cut:].mean()),'recon_mse_mean_rest':float(rec[:cut].mean()),
    'composition':{f'{a}_{b}':int(n) for (a,b),n in v.groupby(['code_module','code_presentation'],observed=True).size().items()}}
o['recon_mse_by_class']={'mean':{'non_dropout':float(rec[y==0].mean()),'dropout':float(rec[y==1].mean())},'median':{'non_dropout':float(np.median(rec[y==0])),'dropout':float(np.median(rec[y==1]))}}
# top dimension SHAP de l'embedding v11 vs descripteurs d'arrêt d'activité
top=pd.read_csv(V11/'shap_top_features.csv').iloc[0]; d=int(top['feature'].split('_')[1])
Xd=np.c_[last,aw,np.log1p(feats['total_clicks'].values)]
o['top_shap_embedding_dim']={'feature':top['feature'],'mean_abs_shap':float(top['mean_abs_shap']),'spearman_with_last_active_week':float(abs(stats.spearmanr(emb[:,d],last)[0])),
    'r2_linear_from_last_active_active_weeks_log_total':float(LinearRegression().fit(Xd,emb[:,d]).score(Xd,emb[:,d]))}
# TOST v11 recalculé avec la variance corrigée de Nadeau & Bengio (k=5)
sc=json.load(open(V11/'statistical_comparison.json')); n=k=5
tc=stats.t.ppf(0.95,n-1); se=(sc['ci90_high']-sc['ci90_low'])/2/tc; se_c=se*np.sqrt((1/k+1/(k-1))*k)
lo,hi=sc['mean_diff']-tc*se_c, sc['mean_diff']+tc*se_c; mg=sc['margin']
p=max(1-stats.t.cdf((sc['mean_diff']+mg)/se_c,n-1),1-stats.t.cdf((mg-sc['mean_diff'])/se_c,n-1))
o['tost_v11_corrected_nadeau_bengio']={'ci90':[float(lo),float(hi)],'p_tost':float(p),'margin':mg,'p_tost_uncorrected':sc['p_tost']}
o['wilcoxon_min_p_n5_two_sided']=2/2**5
json.dump(o,open(OUTJ,'w'),indent=2); print(json.dumps(o,indent=2))
