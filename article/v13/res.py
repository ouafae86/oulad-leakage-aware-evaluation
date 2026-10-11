"""Charge TOUS les résultats (JSON/CSV) et expose un dict N de chiffres prêts à insérer dans le texte."""
import json, numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
import os
ROOT=Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parents[2]))
E=str(ROOT/'results'/'early_windows'); UP=str(ROOT/'results'/'v11_end_of_course')
W=json.load(open(f'{E}/results_windows.json')); A=json.load(open(f'{E}/results_designA.json'))
L=json.load(open(f'{E}/results_labels_w12.json')); X=json.load(open(f'{E}/results_extra_labels.json'))
P=json.load(open(f'{E}/results_lopo_w12.json')); F39=json.load(open(f'{E}/results_extra_w39.json'))
abl=pd.read_csv(f'{UP}/ablation_results.csv'); base=pd.read_csv(f'{UP}/baselines_results.csv',index_col=0)
m=lambda v: float(np.mean(v)); s=lambda v: float(np.std(v))
def paired(a,b):
    d=np.array(a)-np.array(b); k=len(d); se=np.sqrt((1/k+1/(k-1))*d.var(ddof=1)); t=stats.t.ppf(0.975,k-1)
    return float(d.mean()), float(d.mean()-t*se), float(d.mean()+t*se)
def fmt(x,n=3): return f"{x:.{n}f}"
def ci(t): return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
# --- résultats ajoutés par T-05 à T-15 (chargés s'ils existent)
D=str(ROOT/'results'/'diagnostics')
def _load(p): return json.load(open(p)) if os.path.exists(p) else None
S={w:_load(f'{E}/results_seeds_w{w}.json') for w in ['4','8','12','20'] if os.path.exists(f'{E}/results_seeds_w{w}.json')}   # graines de l'autoencodeur
B3P=_load(f'{E}/results_b3prime_w12.json')      # B3' et B3 rejoué
BM=_load(f'{E}/results_by_module.json')         # par module / présentation
BS=_load(f'{E}/results_bootstrap_folds.json')   # bootstrap apparié par étudiant, AUC moyennée sur les plis (D-031) : celui que cite l'article
BSP=_load(f'{E}/results_bootstrap.json')        # même bootstrap sur prédictions regroupées (T-10), conservé pour comparaison
DG=_load(f'{D}/diagnostics_v11.json'); CS=_load(f'{D}/cohort_stats.json')
SC=_load(f'{UP}/statistical_comparison.json')
BPS=_load(f'{E}/results_b3prime_seeds_w12.json')   # pré-entraînement de B3' avec 5 graines d'autoencodeur (T-19)
SA=_load(f'{E}/results_seedavg.json')            # comparaisons en moyenne sur 5 graines d'autoencodeur (T-20)
TT=_load(f'{E}/results_typetotals.json')         # contrôle « totaux par type d'activité » (T-20)
SPT=_load(f'{E}/results_seed_partitions.json')   # graines x partitions (T-20)
BPF=_load(f'{E}/results_b3prime_seeds_folds_w12.json')   # AUC par pli des B3' multi-graines (T-20)
MC=_load(f'{E}/results_by_module_ci.json')     # intervalles par module / présentation (T-18)
PT=_load(f'{E}/results_partitions.json')       # stabilité sur 5 partitions supplémentaires (T-18)
SE=_load(f'{E}/results_sensitivity.json')         # sensibilité à la définition de la cohorte (T-17)
EC=_load(f'{E}/results_endcourse_w39.json')    # cours entier rejoué sur la partition actuelle (T-16) : B1, B2, B3, ablation
W0=_load(f'{E}/origin_folds/results_windows.json'); A0=_load(f'{E}/origin_folds/results_designA.json')   # anciens plis (sensibilité)
def bci(v,n=3): return f"{v['diff']:+.{n}f} [{v['ci95'][0]:+.{n}f}, {v['ci95'][1]:+.{n}f}]"   # différence bootstrap
def seed_stats(w, a='AE embedding (64) + XGBoost', b='C4 untrained LSTM encoder + XGBoost'):
    """Moyennes par graine de a, de b et de a−b (référence exclue) pour la fenêtre w."""
    ks=[k for k in S[w] if k!='reference']; xa=np.array([m(S[w][k][a]['auc']) for k in ks]); xb=np.array([m(S[w][k][b]['auc']) for k in ks])
    return {'n':len(ks),'a':xa,'b':xb,'d':xa-xb}
def b3avg(k): return np.mean([B3P[k][s_]['auc'] for s_ in B3P[k]],axis=0)                      # AUC par pli, moyenné sur les graines
