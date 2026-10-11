#!/usr/bin/env python3
"""Lit results/early_windows/*.json et écrit results/early_windows/SUMMARY.md (tableaux prêts à citer).
   Les chiffres du texte de l'article doivent venir de ce fichier ou de article/v12/res.py, jamais d'une saisie manuelle."""
import json, numpy as np
from pathlib import Path
from scipy import stats
R = Path(__file__).resolve().parents[1]/'results'/'early_windows'
W=json.load(open(R/'results_windows.json')); A=json.load(open(R/'results_designA.json'))
L=json.load(open(R/'results_labels_w12.json')); X=json.load(open(R/'results_extra_labels.json'))
P=json.load(open(R/'results_lopo_w12.json')); F39=json.load(open(R/'results_extra_w39.json'))
m=lambda v: float(np.mean(v)); sd=lambda v: float(np.std(v))
def paired(a,b):
    d=np.array(a)-np.array(b); k=len(d); se=np.sqrt((1/k+1/(k-1))*d.var(ddof=1)); t=stats.t.ppf(0.975,k-1)
    return d.mean(), d.mean()-t*se, d.mean()+t*se
o=["# Synthèse des expériences précoces (générée par tools/summarize_results.py — ne pas éditer à la main)\n"]
o.append("## 1. Ensembles à risque (conception B) et part des retraits déjà enregistrés\n\n| Fenêtre (sem.) | N à risque | Taux de retrait futur | Retraits déjà connus (conception A) |\n|---|---|---|---|")
for w in ['4','8','12','20']: o.append(f"| {w} | {W[w]['N']} | {W[w]['prev']:.3f} | {A[w]['share_already_withdrawn']:.1%} |")
o.append(f"| 39 (cours entier) | {A['39']['N']} (conception A) | {A['39']['prev']:.3f} | {A['39']['share_already_withdrawn']:.1%} |")
o.append("\n## 2. Conception B — AUC-ROC moyen ± écart-type (GroupKFold 5, XGBoost)\n")
keys=['Aggregated(8) [B1-like]','C2 weekly totals','C1 raw sequence flattened','C3 PCA-64','C4 untrained LSTM encoder','AE embedding (64)','Combined (AE + 8 desc.)']
o.append("| Modèle | "+" | ".join(f"{w} sem." for w in ['4','8','12','20'])+" |\n|---|---|---|---|---|")
for k in keys: o.append(f"| {k} | "+" | ".join((f"{m(W[w][k]['auc']):.4f} ± {sd(W[w][k]['auc']):.4f}" if k in W[w] else "—") for w in ['4','8','12','20'])+" |")
o.append("\n## 3. Conception A (tous les étudiants) — AUC-ROC moyen\n\n| Modèle | "+" | ".join(f"{w} sem." for w in ['4','8','12','20','39'])+" |\n|---|---|---|---|---|---|")
for k in ['Aggregated(8)','Weekly totals','Raw sequence','Last active week (1 var.)']: o.append(f"| {k} | "+" | ".join(f"{m(A[w][k]['auc']):.4f}" for w in ['4','8','12','20','39'])+" |")
o.append("\n## 4. Différences appariées d'AUC (conception B), IC95 % corrigé Nadeau & Bengio (k=5, n_test/n_train=1/4)\n\n| Fenêtre | Comparaison | Différence [IC95 %] |\n|---|---|---|")
for w in [w for w in ['4','8','12','20'] if 'AE embedding (64)' in W[w]]:
    for a,b in [('AE embedding (64)','C4 untrained LSTM encoder'),('AE embedding (64)','C3 PCA-64'),('AE embedding (64)','C1 raw sequence flattened'),('Combined (AE + 8 desc.)','Aggregated(8) [B1-like]')]:
        d,lo,hi=paired(W[w][a]['auc'],W[w][b]['auc']); o.append(f"| {w} | {a} − {b} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
d,lo,hi=paired(W['12']['AE embedding (64)']['auc'],L['1.0']['B3 supervised LSTM']['auc']); o.append(f"| 12 | AE+XGB − B3 (100 % étiquettes) | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
o.append("\n## 5. Efficacité en étiquettes, fenêtre 12 (AUC moyen)\n")
fr=['0.01','0.02','0.05','0.1','0.25','1.0']
cols=[('AE + LogReg',lambda f:L[f]['AE + LogReg']['auc']),('AE + XGBoost',lambda f:L[f]['AE + XGBoost']['auc']),('Aggregated(8) + XGBoost',lambda f:L[f]['Aggregated(8) + XGBoost']['auc']),('Raw + XGBoost',lambda f:L[f]['Raw sequence + XGBoost']['auc']),('B3 supervised LSTM',lambda f:L[f].get('B3 supervised LSTM',{}).get('auc')),('PCA-64 + LogReg',lambda f:X[f]['PCA-64 + LogReg']),('Untrained enc. + LogReg',lambda f:X[f]['Untrained LSTM enc. + LogReg']),('Raw + LogReg (C=0.01)',lambda f:X[f]['Raw sequence + LogReg (C=0.01)'])]
o.append("| Étiquettes | "+" | ".join(c for c,_ in cols)+" |\n|---|"+"---|"*len(cols))
for f in fr: o.append(f"| {float(f)*100:g} % | "+" | ".join((f"{m(g(f)):.4f}" if g(f) else "—") for _,g in cols)+" |")
o.append("\n## 6. Présentations retenues (LOPO, fenêtre 12), AUC\n")
mods=[k for k,v in next(iter(P.values())).items() if isinstance(v,dict)]
o.append("| Modèle | "+" | ".join(P.keys())+" | moyenne |\n|---|"+"---|"*(len(P)+1))
for k in mods: o.append(f"| {k} | "+" | ".join(f"{P[p][k]['auc']:.4f}" for p in P)+f" | {np.mean([P[p][k]['auc'] for p in P]):.4f} |")
o.append("\n## 7. Cours entier : autoencodeur de l'article vs encodeur LSTM non entraîné\n\n| Représentation | XGBoost | LogReg |\n|---|---|---|")
for k,v in F39.items(): o.append(f"| {k} | {m(v['xgb']):.4f} | {m(v['lr']):.4f} |")
# --- T-05 : variance de l'autoencodeur entre graines (section ajoutée seulement si le fichier existe)
cols=['AE embedding (64) + XGBoost','AE embedding (64) + LogReg','C4 untrained LSTM encoder + XGBoost','C4 untrained LSTM encoder + LogReg']
SW={w:json.load(open(R/f'results_seeds_w{w}.json')) for w in ['4','8','12','20'] if (R/f'results_seeds_w{w}.json').exists()}
if SW:
    o.append(f"\n## 8. Variance entre graines de l'autoencodeur (fenêtres {', '.join(SW)} sem. ; mêmes plis GroupKFold 5)\n")
    o.append("AUC-ROC moyen sur 5 plis. Seule la graine de l'autoencodeur varie (classifieurs : graine 42). « référence » = artefact `ae_w<w>.npz` utilisé dans les sections 2 à 7, réévalué.")
    sg=lambda v,sign: (f"{v:+.4f}" if sign else f"{v:.4f}")
    synth=[]
    for w,S in SW.items():
        seeds=[k for k in S if k!='reference']
        o.append(f"\n**Fenêtre {w} sem. ({len(seeds)} graines)**\n\n| Graine | Époques | Perte de validation min. | "+" | ".join(cols)+" | AE − C4 (XGBoost) [IC95 %] |\n|---|---|---|"+"---|"*(len(cols)+1))
        def row(k,lab):
            d,lo,hi=paired(S[k][cols[0]]['auc'],S[k][cols[2]]['auc'])
            o.append(f"| {lab} | {S[k]['epochs']} | {S[k]['val_loss_min']:.4f} | "+" | ".join(f"{m(S[k][c]['auc']):.4f}" for c in cols)+f" | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
        for k in seeds: row(k,k)
        if 'reference' in S: row('reference','référence')
        o.append("\n| Statistique entre graines | "+" | ".join(cols)+" | AE − C4 (XGBoost) | AE − C4 (LogReg) |\n|---|"+"---|"*(len(cols)+2))
        M={c:np.array([m(S[k][c]['auc']) for k in seeds]) for c in cols}
        M['dx']=M[cols[0]]-M[cols[2]]; M['dl']=M[cols[1]]-M[cols[3]]; kk=cols+['dx','dl']
        for lab,fn in [('moyenne',np.mean),('écart-type (n−1)',lambda v: np.std(v,ddof=1)),('minimum',np.min),('maximum',np.max)]:
            o.append(f"| {lab} | "+" | ".join(sg(float(fn(M[c])), c in ('dx','dl') and lab!='écart-type (n−1)') for c in kk)+" |")
        o.append("| écart-type moyen entre plis (pour comparaison) | "+" | ".join(f"{np.mean([sd(S[k][c]['auc']) for k in seeds]):.4f}" for c in cols)+" | — | — |")
        synth.append((w,len(seeds),M))
        if 'reference' in S and 'AE embedding (64)' in W[w]:
            rx=m(S['reference'][cols[0]]['auc']); wx=m(W[w]['AE embedding (64)']['auc'])
            o.append(f"\nContrôle de cohérence (même artefact, mêmes plis) : la référence donne AE + XGBoost {rx:.4f} (section 2 : {wx:.4f}, écart {rx-wx:+.4f}).")
    o.append("\n**Synthèse : gain de l'autoencodeur entraîné sur l'encodeur non entraîné de même graine, par fenêtre**\n\n| Fenêtre | Graines | AE − C4 (XGBoost) : moyenne [min ; max] | Graines où AE > C4 (XGBoost) | AE − C4 (LogReg) : moyenne [min ; max] | Graines où AE > C4 (LogReg) |\n|---|---|---|---|---|---|")
    for w,n,M in synth:
        o.append(f"| {w} sem. | {n} | {M['dx'].mean():+.4f} [{M['dx'].min():+.4f} ; {M['dx'].max():+.4f}] | {int((M['dx']>0).sum())}/{n} | {M['dl'].mean():+.4f} [{M['dl'].min():+.4f} ; {M['dl'].max():+.4f}] | {int((M['dl']>0).sum())}/{n} |")
# --- T-06 : B3' (encodeur identique à celui de l'autoencodeur, entraîné de bout en bout) ; section ajoutée seulement si le fichier existe
BP=R/'results_b3prime_w12.json'
if BP.exists():
    B=json.load(open(BP)); names=[k for k,v in B.items() if isinstance(v,dict) and k!='pretrain']
    favg=lambda k: np.mean([B[k][s_]['auc'] for s_ in B[k]],axis=0)          # AUC par pli, moyenné sur les graines
    o.append("\n## 9. B3′ : LSTM supervisé à encodeur identique à celui de l'autoencodeur (128→64), fenêtre 12, GroupKFold 5\n")
    o.append(f"Autoencodeur de pré-entraînement : graine {B['pretrain']['seed']}, {B['pretrain']['epochs']} époques, perte de validation min. {B['pretrain']['val_loss_min']:.4f}. « B3 rerun » = B3 d'origine rejoué sur cette machine. AUC par pli moyenné sur les graines.\n")
    o.append("| Modèle | Graines | AUC-ROC moyen ± écart-type entre plis | Étendue entre graines (moyenne des plis) | Époques (médiane) |\n|---|---|---|---|---|")
    for k in names:
        ms=[m(B[k][s_]['auc']) for s_ in B[k]]; ep=[e for s_ in B[k] for e in B[k][s_].get('epochs',[])]
        o.append(f"| {k} | {len(B[k])} | {m(favg(k)):.4f} ± {sd(favg(k)):.4f} | "+(f"{min(ms):.4f} – {max(ms):.4f}" if len(ms)>1 else "—")+" | "+(f"{np.median(ep):g}" if ep else "—")+" |")
    o.append(f"\nPour mémoire, B3 de l'expérience « étiquettes » (section 5, 100 % des étiquettes, mêmes plis, 1 graine) : {m(L['1.0']['B3 supervised LSTM']['auc']):.4f}.")
    o.append("\nDifférences appariées par pli (IC95 % corrigé Nadeau & Bengio) :\n\n| Comparaison | Différence [IC95 %] |\n|---|---|")
    MP=R/'results_by_module.json'; ext={}                                   # références prises sur les MÊMES plis (T-07)
    if MP.exists():
        MB=json.load(open(MP))['B']['12']['models']
        ext={'Raw sequence + XGBoost':MB['Raw sequence']['fold_auc'],'Aggregated(8) + XGBoost':MB['Aggregated(8)']['fold_auc']}
    get=lambda k: favg(k) if k in B else ext[k]
    for a_,b_ in [("B3' pretrained","B3' scratch"),("B3' pretrained","AE frozen + XGBoost"),("B3' pretrained","AE frozen + LogReg"),("B3' pretrained","B3 rerun"),("B3' scratch","B3 rerun"),("B3' pretrained","Raw sequence + XGBoost"),("B3' scratch","Raw sequence + XGBoost"),("B3' pretrained","Aggregated(8) + XGBoost")]:
        if (a_ in B or a_ in ext) and (b_ in B or b_ in ext):
            d,lo,hi=paired(get(a_),get(b_)); o.append(f"| {a_} − {b_} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
# --- T-07 : résultats par module et par présentation ; section ajoutée seulement si le fichier existe
MP=R/'results_by_module.json'
if MP.exists():
    MM=json.load(open(MP)); fa=lambda v: (f"{v:.4f}" if v is not None else "—")
    o.append("\n## 10. Résultats par module et par présentation (modèle global XGBoost, prédictions hors pli regroupées)\n")
    o.append("AUC-ROC du modèle global calculé dans chaque sous-groupe. « global » = toutes les prédictions hors pli regroupées (diffère légèrement de la moyenne des plis). Détail par couple module × présentation et PR-AUC : `results_by_module.json`.")
    for des,w,lab in [('B','12','Conception B, fenêtre 12 sem. (retraits futurs seulement)'),('B','4','Conception B, fenêtre 4 sem.'),('A','39','Conception A, cours entier (39 sem.)')]:
        mods=MM[des][w]['models']; first=next(iter(mods.values()))
        for by,tit in [('by_module','module'),('by_presentation','présentation')]:
            g=list(first[by]); o.append(f"\n**{lab} — par {tit}**\n\n| Modèle | "+" | ".join(g)+" | global | étendue |\n|---|"+"---|"*(len(g)+2))
            o.append("| *N (retraits)* | "+" | ".join(f"{first[by][x]['n']} ({first[by][x]['n_pos']})" for x in g)+f" | {first['overall']['n']} ({first['overall']['n_pos']}) | |")
            for k,v in mods.items():
                au=[v[by][x]['auc'] for x in g]; ok=[x for x in au if x is not None]
                o.append(f"| {k} | "+" | ".join(fa(x) for x in au)+f" | {fa(v['overall']['auc'])} | {max(ok)-min(ok):.4f} |")
    o.append("\n**Écart entre conceptions par module (séquence brute, AUC)**\n\n| Module | A, 39 sem. | B, 12 sem. | B, 4 sem. |\n|---|---|---|---|")
    for x in MM['A']['39']['models']['Raw sequence']['by_module']:
        o.append(f"| {x} | "+" | ".join(fa(MM[d_][w_]['models']['Raw sequence']['by_module'][x]['auc']) for d_,w_ in [('A','39'),('B','12'),('B','4')])+" |")
# --- sensibilité au découpage en plis (R-17, D-026) : résultats actuels vs anciens résultats conservés dans origin_folds/
OF=R/'origin_folds'
if (OF/'results_windows.json').exists() and (OF/'results_designA.json').exists():
    W0=json.load(open(OF/'results_windows.json')); A0=json.load(open(OF/'results_designA.json'))
    dB=[(w,k,m(W[w][k]['auc'])-m(W0[w][k]['auc'])) for w in ['4','8','12','20'] for k in keys[:4] if k in W[w] and k in W0[w]]   # sans autoencodeur : seules représentations identiques dans les deux exécutions (D-033)
    dA=[(w,k,m(A[w][k]['auc'])-m(A0[w][k]['auc'])) for w in ['4','8','12','20','39'] for k in A[w] if isinstance(A[w][k],dict) and k in A0[w]]
    o.append("\n## 11. Sensibilité au découpage en plis (résultats actuels vs anciens résultats, `origin_folds/`)\n")
    o.append("Toutes les sections de ce document reposent sur une seule partition `GroupKFold(5)` (scikit-learn 1.9.1, déterministe). Les résultats antérieurs avaient été calculés sur une autre machine, dont la partition différait (ordre des groupes de même taille) et n'avait pas été conservée ; ils sont gardés dans `origin_folds/`. Mêmes données, mêmes représentations, mêmes hyperparamètres ; la partition change (la version de XGBoost de la machine d'origine n'a pas été enregistrée et peut aussi contribuer) :\n")
    o.append("| Conception | Comparaisons | Écart absolu moyen de l'AUC moyen | Écart absolu maximal | Cas le plus éloigné |\n|---|---|---|---|---|")
    for lab,dd in [('B (retraits futurs)',dB),('A (tous les étudiants)',dA)]:
        wmax=max(dd,key=lambda t: abs(t[2])); o.append(f"| {lab} | {len(dd)} | {np.mean([abs(t[2]) for t in dd]):.4f} | {abs(wmax[2]):.4f} | {wmax[1]}, {wmax[0]} sem. ({wmax[2]:+.4f}) |")
    sdB=[sd(W[w][k]['auc']) for w in ['4','8','12','20'] for k in keys if k in W[w]]
    o.append(f"\nPour comparaison, l'écart-type entre plis d'un même modèle va de {min(sdB):.4f} à {max(sdB):.4f} en conception B (section 2).")
# --- T-10 : bootstrap apparié par étudiant sur les prédictions hors pli ; section ajoutée seulement si le fichier existe
for BS, _num, _stat in [(R/'results_bootstrap.json', '12', 'AUC sur les prédictions hors pli regroupées'), (R/'results_bootstrap_folds.json', '13', 'AUC calculée dans chaque pli puis moyennée sur les plis : statistique citée dans l\'article, D-031')]:
    if not BS.exists(): continue
    Q=json.load(open(BS)); nb=Q['meta']['n_boot']
    cb=lambda v: f"{v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
    o.append(f"\n## {_num}. Bootstrap apparié par étudiant ({nb} rééchantillons ; {_stat})\n")
    o.append("Les étudiants (toutes leurs inscriptions ensemble) sont rééchantillonnés avec remise ; différence d'AUC-ROC entre deux modèles évalués sur le même rééchantillon ; IC95 % par percentiles. L'intervalle reflète l'incertitude d'échantillonnage des étudiants, les modèles étant fixés : il n'inclut pas la variabilité d'entraînement (graine : section 8 ; partition : section 11). Ces intervalles remplacent ceux de la section 4, qui ne reposent que sur 5 valeurs par modèle.\n")
    compsB=[c for c in Q['B']['12']['comparisons'] if c in Q['B']['4']['comparisons']]
    o.append("**Conception B (retraits futurs)**\n\n| Comparaison | "+" | ".join(f"{w} sem." for w in ['4','8','12','20'])+" |\n|---|---|---|---|---|")
    for c in compsB: o.append(f"| {c} | "+" | ".join(cb(Q['B'][w]['comparisons'][c]) for w in ['4','8','12','20'])+" |")
    only12=[c for c in Q['B']['12']['comparisons'] if c not in compsB]
    if only12:
        o.append("\n**Conception B, 12 sem. — modèles supervisés (moyenne des graines)**\n\n| Comparaison | Différence [IC95 %] | Part des rééchantillons > 0 |\n|---|---|---|")
        for c in only12: o.append(f"| {c} | {cb(Q['B']['12']['comparisons'][c])} | {Q['B']['12']['comparisons'][c]['share_positive']:.1%} |")
    compsA=list(Q['A']['4']['comparisons'])
    o.append("\n**Conception A (tous les étudiants)**\n\n| Comparaison | "+" | ".join(f"{w} sem." for w in ['4','8','12','20','39'])+" |\n|---|---|---|---|---|---|")
    for c in compsA: o.append(f"| {c} | "+" | ".join(cb(Q['A'][w]['comparisons'][c]) for w in ['4','8','12','20','39'])+" |")
    extra=[c for c in Q['A']['39']['comparisons'] if c not in compsA]
    if extra:
        o.append("\n**Conception A, 39 sem. — embedding v11**\n\n| Comparaison | Différence [IC95 %] |\n|---|---|")
        for c in extra: o.append(f"| {c} | {cb(Q['A']['39']['comparisons'][c])} |")
    o.append("\n**AUC par modèle, conception B, 12 sem. [IC95 % bootstrap]**\n\n| Modèle | Graines | AUC [IC95 %] |\n|---|---|---|")
    for k,v in Q['B']['12']['models'].items(): o.append(f"| {k} | {v['n_seeds']} | {v['auc']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] |")
# --- T-17 : sensibilité à la définition de la cohorte ; section ajoutée seulement si le fichier existe
SEP=R/'results_sensitivity.json'
if SEP.exists():
    SE=json.load(open(SEP)); I=SE['include_no_vle']; XD=SE['exclude_no_date']
    o.append("\n## 14. Sensibilité à la définition de la cohorte (XGBoost, AUC-ROC moyen sur 5 plis)\n")
    o.append(f"**S1 — inscriptions sans aucune activité incluses** ({I['n_added']} inscriptions ajoutées, dont {I['n_added_withdrawn']} retraits ; plis recalculés sur la cohorte étendue)\n\n| Conception, fenêtre | N | dont ajoutées (retraits) | Modèle | Cohorte étendue | Cohorte d'analyse seule, mêmes prédictions |\n|---|---|---|---|---|---|")
    for key,lab in [('A39','A, 39 sem.'),('B12','B, 12 sem.')]:
        for k,v in I[key]['models'].items(): o.append(f"| {lab} | {I[key]['N']} | {I[key]['n_added_in_set']} ({I[key]['n_added_withdrawn_in_set']}) | {k} | {v['extended']['auc']:.4f} | {v['analytic_cohort_only']['auc']:.4f} |")
    o.append(f"\n**S2 — {XD['n_no_date']} retraits sans date de désinscription retirés des ensembles à risque** (conception B, plis de l'analyse principale)\n\n| Fenêtre | Retraits futurs (analyse principale) | Modèle | Analyse principale | Sans ces retraits | Écart |\n|---|---|---|---|---|---|")
    for w in ['4','8','12','20']:
        for k,v in XD[w]['models'].items(): o.append(f"| {w} | {XD[w]['n_pos_main']} | {k} | {v['main']['auc']:.4f} | {v['without_no_date']['auc']:.4f} | {v['without_no_date']['auc']-v['main']['auc']:+.4f} |")
# --- T-19 : pré-entraînement de B3' avec plusieurs graines d'autoencodeur ; section ajoutée seulement si le fichier existe
BPSP=R/'results_b3prime_seeds_w12.json'
if BPSP.exists():
    Q5=json.load(open(BPSP)); gv=lambda v: f"{v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
    o.append(f"\n## 15. Pré-entraînement de B3′ selon la graine de l'autoencodeur (12 sem., {len(Q5['meta']['supervised_seeds'])} graines supervisées par autoencodeur ; bootstrap en moyenne des plis)\n")
    o.append(f"B3′ entraîné de zéro : {Q5['scratch']['auc']:.4f}.\n\n| Graine de l'autoencodeur | Perte de validation min. de l'autoencodeur | B3′ pré-entraîné (AUC moyen) | Gain sur B3′ de zéro [IC95 %] |\n|---|---|---|---|")
    for a_,v in Q5['ae'].items(): o.append(f"| {a_} | {v['val_loss_min']:.4f} | {v['auc']:.4f} | {gv(v['gain_vs_scratch'])} |")
    o.append(f"| toutes ({Q5['all_ae_seeds']['n_runs']} modèles) | | {Q5['all_ae_seeds']['auc']:.4f} | {gv(Q5['all_ae_seeds']['gain_vs_scratch'])} |")
    o.append(f"\nB3′ pré-entraîné (toutes graines) − séquence brute + XGBoost : {gv(Q5['all_ae_seeds']['vs_raw_sequence'])}.")
# --- T-20 : moyenne sur 5 graines, contrôle par type d'activité, graines x partitions ; sections ajoutées si les fichiers existent
gq=lambda v: f"{v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
if (R/'results_seedavg.json').exists():
    SAV=json.load(open(R/'results_seedavg.json'))
    o.append("\n## 16. Comparaisons de l'autoencodeur en moyenne sur 5 graines (conception B, bootstrap en moyenne des plis)\n\n| Comparaison | 4 sem. | 8 sem. | 12 sem. | 20 sem. |\n|---|---|---|---|---|")
    for c in SAV['4']['comparisons']: o.append(f"| {c} | "+" | ".join(gq(SAV[w]['comparisons'][c]) for w in ['4','8','12','20'])+" |")
if (R/'results_typetotals.json').exists():
    TTV=json.load(open(R/'results_typetotals.json'))
    o.append("\n## 17. Contrôle « totaux par type d'activité » (C5) et « totaux hebdomadaires » (C2), conception B\n\n| Modèle ou comparaison | 4 sem. | 8 sem. | 12 sem. | 20 sem. |\n|---|---|---|---|---|")
    for k in TTV['4']['models']: o.append(f"| AUC, {k} | "+" | ".join(f"{TTV[w]['models'][k]['auc']:.4f}" for w in ['4','8','12','20'])+" |")
    for c in TTV['4']['comparisons']: o.append(f"| {c} | "+" | ".join(gq(TTV[w]['comparisons'][c]) for w in ['4','8','12','20'])+" |")
if (R/'results_seed_partitions.json').exists():
    SPV=json.load(open(R/'results_seed_partitions.json')); mm=lambda d: f"{d.mean():+.4f} [{d.min():+.4f}, {d.max():+.4f}]"
    o.append(f"\n## 18. Graines × partitions : différences sur {SPV['meta']['n_partitions']} partitions, embedding et encodeur non entraîné moyennés sur 5 graines (moyenne [min, max])\n\n| Comparaison | 4 sem. | 8 sem. | 12 sem. | 20 sem. |\n|---|---|---|---|---|")
    for lab,b in [('Embedding − encodeur non entraîné','C4 untrained'),('Embedding − PCA-64','C3 PCA-64'),('Embedding − séquence brute','Raw sequence')]:
        o.append(f"| {lab} | "+" | ".join(mm(np.array(SPV[w]['Embedding']).mean(1)-(np.array(SPV[w][b]).mean(1) if b=='C4 untrained' else np.array(SPV[w][b]))) for w in ['4','8','12','20'])+" |")
if (R/'results_b3prime_seeds_folds_w12.json').exists() and (R/'results_b3prime_seeds_w12.json').exists():
    BF=json.load(open(R/'results_b3prime_seeds_folds_w12.json'))
    o.append("\n## 19. Pré-entraînement de B3′ : intervalle fondé sur les 5 plis (variance corrigée de Nadeau & Bengio)\n\n| Graine de l'autoencodeur | Gain sur B3′ de zéro [IC95 %] |\n|---|---|")
    for a_ in list(BF['ae'])+['all_ae_seeds']:
        f_=(BF['ae'][a_] if a_ in BF['ae'] else BF[a_])['fold_auc']; d,lo,hi=paired(f_,BF['scratch']['fold_auc']); o.append(f"| {a_ if a_ in BF['ae'] else 'toutes'} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
(R/'SUMMARY.md').write_text("\n".join(o)+"\n"); print("SUMMARY.md écrit")
