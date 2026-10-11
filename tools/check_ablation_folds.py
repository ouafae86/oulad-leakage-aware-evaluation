#!/usr/bin/env python3
"""T-09 / K-11 : vérifie que results/v11_end_of_course/ablation_fold_aucs.json (AUC par pli des 9 configurations v11)
   est cohérent avec les autres résultats v11, SANS rien recalculer ni écrire (lecture seule).
   1. copie identique à interim/ablation_fold_aucs.json (si présent) ;
   2. moyenne et écart-type par configuration = ablation_results.csv (tolérance 0,002 ; critère de T-09) ;
   3. B1 de baselines_results.csv = Aggregated/RandomForest pli à pli (même partition GroupKFold) ;
   4. Wilcoxon et TOST recalculés à partir des AUC par pli = statistical_comparison.json.
   Usage : python tools/check_ablation_folds.py     (code de sortie 1 si un contrôle échoue)"""
import ast, hashlib, json, sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon, t as t_dist
ROOT = Path(__file__).resolve().parents[1]
V = ROOT/'results'/'v11_end_of_course'
TOL = 0.002
fails = 0
def report(ok, msg):
    global fails
    fails += (not ok); print(('[OK  ] ' if ok else '[FAIL] ') + msg)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

F = json.load(open(V/'ablation_fold_aucs.json'))
abl = pd.read_csv(V/'ablation_results.csv')
base = pd.read_csv(V/'baselines_results.csv', index_col=0)
stat = json.load(open(V/'statistical_comparison.json'))

# 1. même fichier que la copie locale de l'ancien interim/
src = ROOT/'interim'/'ablation_fold_aucs.json'
if src.exists(): report(sha(src) == sha(V/'ablation_fold_aucs.json'), "results/ et interim/ : ablation_fold_aucs.json identiques (SHA-256)")
else: print('[INFO] interim/ablation_fold_aucs.json absent : contrôle 1 ignoré')

# 2. moyennes et écarts-types vs ablation_results.csv
report(len(F) == len(abl) == 9 and all(len(v) == 5 for v in F.values()), f"{len(F)} configurations x 5 plis")
dm = ds = 0.0
for _, r in abl.iterrows():
    a = np.array(F[f"{r['Config']}__{r['Model']}"])
    dm = max(dm, abs(a.mean() - r['AUC-ROC'])); ds = max(ds, abs(a.std() - r['AUC_std']))
report(dm < TOL and ds < TOL, f"écart maximal avec ablation_results.csv : moyenne {dm:.2e}, écart-type {ds:.2e} (tolérance {TOL})")

# 3. même partition que les baselines : B1 = Aggregated / RandomForest
b1 = np.array(ast.literal_eval(base.loc['B1_RandomForest', 'auc_folds']))
d1 = float(np.abs(b1 - np.array(F['Aggregated__RandomForest'])).max())
report(d1 < 1e-9, f"B1 (baselines_results.csv) = Aggregated/RandomForest pli à pli (écart maximal {d1:.2e})")

# 4. tests statistiques recalculés à partir des AUC par pli
best = max(F, key=lambda k: np.mean(F[k])); a = np.array(F[best])
b3 = np.array(ast.literal_eval(base.loc['B3_LSTM_supervised', 'auc_folds']))
cfg, mdl = best.split('__')
report(stat['best_config'] == str((cfg, mdl)), f"meilleure configuration recalculée = {best} (fichier : {stat['best_config']})")
margin = float(base['auc_mean'].sort_values().diff().dropna().min())
d = a - b3; n = len(d); se = d.std(ddof=1)/np.sqrt(n)
p_tost = max(1 - t_dist.cdf((d.mean() + margin)/se, df=n-1), 1 - t_dist.cdf((margin - d.mean())/se, df=n-1))
lo, hi = t_dist.interval(0.90, df=n-1, loc=d.mean(), scale=se)
p_w = wilcoxon(a, b3)[1]
for name, got, exp in [('différence moyenne vs B3', d.mean(), stat['mean_diff']), ("marge d'équivalence", margin, stat['equivalence_margin']),
                       ('p Wilcoxon', p_w, stat['p_wilcoxon']), ('p TOST', p_tost, stat['p_tost']),
                       ('IC90 % bas', lo, stat['ci90_low']), ('IC90 % haut', hi, stat['ci90_high'])]:
    report(abs(got - exp) < 1e-9, f"{name} : recalculé {got:.6f}, fichier {exp:.6f}")

print(f"\nRésumé : {'0 échec' if not fails else str(fails) + ' échec(s)'}")
sys.exit(1 if fails else 0)
