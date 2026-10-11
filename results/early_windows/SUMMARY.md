# Synthèse des expériences précoces (générée par tools/summarize_results.py — ne pas éditer à la main)

## 1. Ensembles à risque (conception B) et part des retraits déjà enregistrés

| Fenêtre (sem.) | N à risque | Taux de retrait futur | Retraits déjà connus (conception A) |
|---|---|---|---|
| 4 | 27041 | 0.184 | 30.4% |
| 8 | 26067 | 0.154 | 44.0% |
| 12 | 25294 | 0.128 | 54.8% |
| 20 | 23876 | 0.076 | 74.6% |
| 39 (cours entier) | 29228 (conception A) | 0.245 | 99.0% |

## 2. Conception B — AUC-ROC moyen ± écart-type (GroupKFold 5, XGBoost)

| Modèle | 4 sem. | 8 sem. | 12 sem. | 20 sem. |
|---|---|---|---|---|
| Aggregated(8) [B1-like] | 0.5799 ± 0.0094 | 0.5970 ± 0.0095 | 0.6060 ± 0.0078 | 0.6474 ± 0.0150 |
| C2 weekly totals | 0.5543 ± 0.0119 | 0.5929 ± 0.0058 | 0.5999 ± 0.0090 | 0.6769 ± 0.0064 |
| C1 raw sequence flattened | 0.6207 ± 0.0100 | 0.6440 ± 0.0030 | 0.6471 ± 0.0079 | 0.7023 ± 0.0161 |
| C3 PCA-64 | 0.6088 ± 0.0056 | 0.6356 ± 0.0047 | 0.6335 ± 0.0078 | 0.6904 ± 0.0087 |
| C4 untrained LSTM encoder | 0.6118 ± 0.0120 | 0.6178 ± 0.0094 | 0.6278 ± 0.0081 | 0.6911 ± 0.0123 |
| AE embedding (64) | 0.6162 ± 0.0102 | 0.6366 ± 0.0111 | 0.6381 ± 0.0096 | 0.6842 ± 0.0115 |
| Combined (AE + 8 desc.) | 0.6181 ± 0.0125 | 0.6413 ± 0.0087 | 0.6405 ± 0.0135 | 0.6823 ± 0.0091 |

## 3. Conception A (tous les étudiants) — AUC-ROC moyen

| Modèle | 4 sem. | 8 sem. | 12 sem. | 20 sem. | 39 sem. |
|---|---|---|---|---|---|
| Aggregated(8) | 0.6659 | 0.7378 | 0.7753 | 0.8466 | 0.9100 |
| Weekly totals | 0.6492 | 0.7416 | 0.7851 | 0.8743 | 0.9239 |
| Raw sequence | 0.6989 | 0.7709 | 0.8028 | 0.8793 | 0.9268 |
| Last active week (1 var.) | 0.6221 | 0.7156 | 0.7674 | 0.8518 | 0.9174 |

## 4. Différences appariées d'AUC (conception B), IC95 % corrigé Nadeau & Bengio (k=5, n_test/n_train=1/4)

| Fenêtre | Comparaison | Différence [IC95 %] |
|---|---|---|
| 4 | AE embedding (64) − C4 untrained LSTM encoder | +0.0044 [-0.0138, +0.0226] |
| 4 | AE embedding (64) − C3 PCA-64 | +0.0074 [-0.0044, +0.0192] |
| 4 | AE embedding (64) − C1 raw sequence flattened | -0.0045 [-0.0134, +0.0044] |
| 4 | Combined (AE + 8 desc.) − Aggregated(8) [B1-like] | +0.0382 [+0.0234, +0.0529] |
| 8 | AE embedding (64) − C4 untrained LSTM encoder | +0.0187 [+0.0096, +0.0279] |
| 8 | AE embedding (64) − C3 PCA-64 | +0.0009 [-0.0190, +0.0208] |
| 8 | AE embedding (64) − C1 raw sequence flattened | -0.0074 [-0.0307, +0.0158] |
| 8 | Combined (AE + 8 desc.) − Aggregated(8) [B1-like] | +0.0443 [+0.0173, +0.0714] |
| 12 | AE embedding (64) − C4 untrained LSTM encoder | +0.0103 [-0.0170, +0.0377] |
| 12 | AE embedding (64) − C3 PCA-64 | +0.0047 [-0.0136, +0.0229] |
| 12 | AE embedding (64) − C1 raw sequence flattened | -0.0089 [-0.0348, +0.0169] |
| 12 | Combined (AE + 8 desc.) − Aggregated(8) [B1-like] | +0.0345 [+0.0130, +0.0559] |
| 20 | AE embedding (64) − C4 untrained LSTM encoder | -0.0069 [-0.0295, +0.0156] |
| 20 | AE embedding (64) − C3 PCA-64 | -0.0062 [-0.0207, +0.0083] |
| 20 | AE embedding (64) − C1 raw sequence flattened | -0.0181 [-0.0412, +0.0049] |
| 20 | Combined (AE + 8 desc.) − Aggregated(8) [B1-like] | +0.0349 [+0.0069, +0.0628] |
| 12 | AE+XGB − B3 (100 % étiquettes) | -0.0060 [-0.0353, +0.0233] |

## 5. Efficacité en étiquettes, fenêtre 12 (AUC moyen)

| Étiquettes | AE + LogReg | AE + XGBoost | Aggregated(8) + XGBoost | Raw + XGBoost | B3 supervised LSTM | PCA-64 + LogReg | Untrained enc. + LogReg | Raw + LogReg (C=0.01) |
|---|---|---|---|---|---|---|---|---|
| 1 % | 0.5577 | 0.5432 | 0.5220 | 0.5411 | 0.5131 | 0.5423 | 0.5659 | 0.5755 |
| 2 % | 0.5661 | 0.5534 | 0.5352 | 0.5559 | — | 0.5651 | 0.5749 | 0.5871 |
| 5 % | 0.5836 | 0.5794 | 0.5522 | 0.5823 | 0.5606 | 0.5885 | 0.5926 | 0.5968 |
| 10 % | 0.6072 | 0.5901 | 0.5620 | 0.6029 | — | 0.6119 | 0.6060 | 0.6067 |
| 25 % | 0.6278 | 0.6099 | 0.5775 | 0.6141 | 0.6216 | 0.6319 | 0.6212 | 0.6234 |
| 100 % | 0.6423 | 0.6381 | 0.6060 | 0.6471 | 0.6442 | 0.6462 | 0.6344 | 0.6434 |

## 6. Présentations retenues (LOPO, fenêtre 12), AUC

| Modèle | 2013B | 2013J | 2014B | 2014J | moyenne |
|---|---|---|---|---|---|
| AE + LogReg | 0.5895 | 0.6411 | 0.6474 | 0.5693 | 0.6118 |
| AE + XGBoost | 0.5957 | 0.6288 | 0.6443 | 0.5982 | 0.6167 |
| Combined (AE+8) + XGBoost | 0.5993 | 0.6239 | 0.6348 | 0.5906 | 0.6122 |
| Aggregated(8) + XGBoost | 0.6110 | 0.5868 | 0.5951 | 0.5860 | 0.5947 |
| Raw sequence + XGBoost | 0.6258 | 0.6257 | 0.6489 | 0.6335 | 0.6335 |
| B3 supervised LSTM | 0.6027 | 0.6312 | 0.6359 | 0.6033 | 0.6183 |

## 7. Cours entier : autoencodeur de l'article vs encodeur LSTM non entraîné

| Représentation | XGBoost | LogReg |
|---|---|---|
| AE (article) | 0.9232 | 0.9202 |
| Untrained LSTM encoder | 0.9236 | 0.9108 |

## 8. Variance entre graines de l'autoencodeur (fenêtres 4, 8, 12, 20 sem. ; mêmes plis GroupKFold 5)

AUC-ROC moyen sur 5 plis. Seule la graine de l'autoencodeur varie (classifieurs : graine 42). « référence » = artefact `ae_w<w>.npz` utilisé dans les sections 2 à 7, réévalué.

**Fenêtre 4 sem. (5 graines)**

| Graine | Époques | Perte de validation min. | AE embedding (64) + XGBoost | AE embedding (64) + LogReg | C4 untrained LSTM encoder + XGBoost | C4 untrained LSTM encoder + LogReg | AE − C4 (XGBoost) [IC95 %] |
|---|---|---|---|---|---|---|---|
| 42 | 30 | 0.0471 | 0.6162 | 0.6191 | 0.6118 | 0.6108 | +0.0044 [-0.0138, +0.0226] |
| 43 | 30 | 0.0590 | 0.6166 | 0.6144 | 0.6080 | 0.6122 | +0.0086 [+0.0023, +0.0150] |
| 44 | 30 | 0.0545 | 0.6182 | 0.6178 | 0.6068 | 0.6061 | +0.0114 [-0.0149, +0.0378] |
| 45 | 30 | 0.0487 | 0.6098 | 0.6170 | 0.6101 | 0.6074 | -0.0003 [-0.0204, +0.0198] |
| 46 | 30 | 0.0630 | 0.6135 | 0.6192 | 0.6132 | 0.6075 | +0.0003 [-0.0056, +0.0061] |
| référence | 30 | 0.0471 | 0.6162 | 0.6191 | 0.6118 | 0.6108 | +0.0044 [-0.0138, +0.0226] |

| Statistique entre graines | AE embedding (64) + XGBoost | AE embedding (64) + LogReg | C4 untrained LSTM encoder + XGBoost | C4 untrained LSTM encoder + LogReg | AE − C4 (XGBoost) | AE − C4 (LogReg) |
|---|---|---|---|---|---|---|
| moyenne | 0.6149 | 0.6175 | 0.6100 | 0.6088 | +0.0049 | +0.0087 |
| écart-type (n−1) | 0.0033 | 0.0020 | 0.0026 | 0.0026 | 0.0051 | 0.0039 |
| minimum | 0.6098 | 0.6144 | 0.6068 | 0.6061 | -0.0003 | +0.0022 |
| maximum | 0.6182 | 0.6192 | 0.6132 | 0.6122 | +0.0114 | +0.0117 |
| écart-type moyen entre plis (pour comparaison) | 0.0104 | 0.0087 | 0.0100 | 0.0102 | — | — |

Contrôle de cohérence (même artefact, mêmes plis) : la référence donne AE + XGBoost 0.6162 (section 2 : 0.6162, écart +0.0000).

**Fenêtre 8 sem. (5 graines)**

| Graine | Époques | Perte de validation min. | AE embedding (64) + XGBoost | AE embedding (64) + LogReg | C4 untrained LSTM encoder + XGBoost | C4 untrained LSTM encoder + LogReg | AE − C4 (XGBoost) [IC95 %] |
|---|---|---|---|---|---|---|---|
| 42 | 30 | 0.4807 | 0.6366 | 0.6465 | 0.6178 | 0.6334 | +0.0187 [+0.0096, +0.0279] |
| 43 | 30 | 0.1716 | 0.6319 | 0.6408 | 0.6233 | 0.6270 | +0.0086 [-0.0142, +0.0313] |
| 44 | 30 | 0.1776 | 0.6366 | 0.6437 | 0.6188 | 0.6349 | +0.0178 [+0.0060, +0.0296] |
| 45 | 30 | 0.1905 | 0.6350 | 0.6406 | 0.6244 | 0.6321 | +0.0106 [-0.0026, +0.0238] |
| 46 | 30 | 0.1911 | 0.6374 | 0.6420 | 0.6183 | 0.6317 | +0.0190 [+0.0036, +0.0345] |
| référence | 30 | 0.4807 | 0.6366 | 0.6465 | 0.6178 | 0.6334 | +0.0187 [+0.0096, +0.0279] |

| Statistique entre graines | AE embedding (64) + XGBoost | AE embedding (64) + LogReg | C4 untrained LSTM encoder + XGBoost | C4 untrained LSTM encoder + LogReg | AE − C4 (XGBoost) | AE − C4 (LogReg) |
|---|---|---|---|---|---|---|
| moyenne | 0.6355 | 0.6427 | 0.6205 | 0.6318 | +0.0149 | +0.0109 |
| écart-type (n−1) | 0.0022 | 0.0024 | 0.0031 | 0.0030 | 0.0050 | 0.0024 |
| minimum | 0.6319 | 0.6406 | 0.6178 | 0.6270 | +0.0086 | +0.0085 |
| maximum | 0.6374 | 0.6465 | 0.6244 | 0.6349 | +0.0190 | +0.0138 |
| écart-type moyen entre plis (pour comparaison) | 0.0104 | 0.0103 | 0.0091 | 0.0081 | — | — |

Contrôle de cohérence (même artefact, mêmes plis) : la référence donne AE + XGBoost 0.6366 (section 2 : 0.6366, écart +0.0000).

**Fenêtre 12 sem. (5 graines)**

| Graine | Époques | Perte de validation min. | AE embedding (64) + XGBoost | AE embedding (64) + LogReg | C4 untrained LSTM encoder + XGBoost | C4 untrained LSTM encoder + LogReg | AE − C4 (XGBoost) [IC95 %] |
|---|---|---|---|---|---|---|---|
| 42 | 28 | 0.3079 | 0.6381 | 0.6423 | 0.6278 | 0.6344 | +0.0103 [-0.0170, +0.0377] |
| 43 | 30 | 0.2905 | 0.6371 | 0.6420 | 0.6329 | 0.6391 | +0.0042 [-0.0180, +0.0264] |
| 44 | 30 | 0.2783 | 0.6360 | 0.6471 | 0.6303 | 0.6363 | +0.0058 [+0.0001, +0.0114] |
| 45 | 30 | 0.2711 | 0.6343 | 0.6470 | 0.6363 | 0.6328 | -0.0019 [-0.0147, +0.0108] |
| 46 | 30 | 0.2536 | 0.6373 | 0.6418 | 0.6288 | 0.6361 | +0.0085 [-0.0048, +0.0219] |
| référence | 28 | 0.3079 | 0.6381 | 0.6423 | 0.6278 | 0.6344 | +0.0103 [-0.0170, +0.0377] |

| Statistique entre graines | AE embedding (64) + XGBoost | AE embedding (64) + LogReg | C4 untrained LSTM encoder + XGBoost | C4 untrained LSTM encoder + LogReg | AE − C4 (XGBoost) | AE − C4 (LogReg) |
|---|---|---|---|---|---|---|
| moyenne | 0.6366 | 0.6441 | 0.6312 | 0.6357 | +0.0054 | +0.0083 |
| écart-type (n−1) | 0.0015 | 0.0028 | 0.0034 | 0.0024 | 0.0047 | 0.0044 |
| minimum | 0.6343 | 0.6418 | 0.6278 | 0.6328 | -0.0019 | +0.0030 |
| maximum | 0.6381 | 0.6471 | 0.6363 | 0.6391 | +0.0103 | +0.0142 |
| écart-type moyen entre plis (pour comparaison) | 0.0072 | 0.0086 | 0.0079 | 0.0075 | — | — |

Contrôle de cohérence (même artefact, mêmes plis) : la référence donne AE + XGBoost 0.6381 (section 2 : 0.6381, écart +0.0000).

**Fenêtre 20 sem. (5 graines)**

| Graine | Époques | Perte de validation min. | AE embedding (64) + XGBoost | AE embedding (64) + LogReg | C4 untrained LSTM encoder + XGBoost | C4 untrained LSTM encoder + LogReg | AE − C4 (XGBoost) [IC95 %] |
|---|---|---|---|---|---|---|---|
| 42 | 21 | 0.4511 | 0.6842 | 0.7025 | 0.6911 | 0.7045 | -0.0069 [-0.0295, +0.0156] |
| 43 | 30 | 0.5640 | 0.6900 | 0.7073 | 0.6891 | 0.7055 | +0.0009 [-0.0147, +0.0166] |
| 44 | 30 | 0.4066 | 0.6865 | 0.7054 | 0.6893 | 0.7039 | -0.0028 [-0.0186, +0.0130] |
| 45 | 20 | 0.4196 | 0.6791 | 0.7022 | 0.6963 | 0.7012 | -0.0172 [-0.0314, -0.0030] |
| 46 | 24 | 0.4251 | 0.6865 | 0.7059 | 0.6944 | 0.7022 | -0.0079 [-0.0319, +0.0161] |
| référence | 21 | 0.4511 | 0.6842 | 0.7025 | 0.6911 | 0.7045 | -0.0069 [-0.0295, +0.0156] |

| Statistique entre graines | AE embedding (64) + XGBoost | AE embedding (64) + LogReg | C4 untrained LSTM encoder + XGBoost | C4 untrained LSTM encoder + LogReg | AE − C4 (XGBoost) | AE − C4 (LogReg) |
|---|---|---|---|---|---|---|
| moyenne | 0.6852 | 0.7046 | 0.6920 | 0.7035 | -0.0068 | +0.0012 |
| écart-type (n−1) | 0.0040 | 0.0022 | 0.0032 | 0.0017 | 0.0068 | 0.0020 |
| minimum | 0.6791 | 0.7022 | 0.6891 | 0.7012 | -0.0172 | -0.0020 |
| maximum | 0.6900 | 0.7073 | 0.6963 | 0.7055 | +0.0009 | +0.0036 |
| écart-type moyen entre plis (pour comparaison) | 0.0120 | 0.0139 | 0.0151 | 0.0145 | — | — |

Contrôle de cohérence (même artefact, mêmes plis) : la référence donne AE + XGBoost 0.6842 (section 2 : 0.6842, écart +0.0000).

**Synthèse : gain de l'autoencodeur entraîné sur l'encodeur non entraîné de même graine, par fenêtre**

| Fenêtre | Graines | AE − C4 (XGBoost) : moyenne [min ; max] | Graines où AE > C4 (XGBoost) | AE − C4 (LogReg) : moyenne [min ; max] | Graines où AE > C4 (LogReg) |
|---|---|---|---|---|---|
| 4 sem. | 5 | +0.0049 [-0.0003 ; +0.0114] | 4/5 | +0.0087 [+0.0022 ; +0.0117] | 5/5 |
| 8 sem. | 5 | +0.0149 [+0.0086 ; +0.0190] | 5/5 | +0.0109 [+0.0085 ; +0.0138] | 5/5 |
| 12 sem. | 5 | +0.0054 [-0.0019 ; +0.0103] | 4/5 | +0.0083 [+0.0030 ; +0.0142] | 5/5 |
| 20 sem. | 5 | -0.0068 [-0.0172 ; +0.0009] | 1/5 | +0.0012 [-0.0020 ; +0.0036] | 4/5 |

## 9. B3′ : LSTM supervisé à encodeur identique à celui de l'autoencodeur (128→64), fenêtre 12, GroupKFold 5

Autoencodeur de pré-entraînement : graine 42, 28 époques, perte de validation min. 0.3079. « B3 rerun » = B3 d'origine rejoué sur cette machine. AUC par pli moyenné sur les graines.

| Modèle | Graines | AUC-ROC moyen ± écart-type entre plis | Étendue entre graines (moyenne des plis) | Époques (médiane) |
|---|---|---|---|---|
| AE frozen + XGBoost | 1 | 0.6381 ± 0.0096 | — | — |
| AE frozen + LogReg | 1 | 0.6423 ± 0.0124 | — | — |
| B3' pretrained | 3 | 0.6514 ± 0.0097 | 0.6497 – 0.6547 | 7 |
| B3' scratch | 3 | 0.6455 ± 0.0086 | 0.6403 – 0.6489 | 9 |
| B3 rerun | 3 | 0.6445 ± 0.0087 | 0.6425 – 0.6474 | — |

Pour mémoire, B3 de l'expérience « étiquettes » (section 5, 100 % des étiquettes, mêmes plis, 1 graine) : 0.6442.

Différences appariées par pli (IC95 % corrigé Nadeau & Bengio) :

| Comparaison | Différence [IC95 %] |
|---|---|
| B3' pretrained − B3' scratch | +0.0059 [+0.0003, +0.0115] |
| B3' pretrained − AE frozen + XGBoost | +0.0133 [-0.0144, +0.0409] |
| B3' pretrained − AE frozen + LogReg | +0.0091 [-0.0028, +0.0211] |
| B3' pretrained − B3 rerun | +0.0069 [-0.0066, +0.0204] |
| B3' scratch − B3 rerun | +0.0010 [-0.0103, +0.0122] |
| B3' pretrained − Raw sequence + XGBoost | +0.0043 [-0.0062, +0.0148] |
| B3' scratch − Raw sequence + XGBoost | -0.0016 [-0.0123, +0.0091] |
| B3' pretrained − Aggregated(8) + XGBoost | +0.0454 [+0.0330, +0.0577] |

## 10. Résultats par module et par présentation (modèle global XGBoost, prédictions hors pli regroupées)

AUC-ROC du modèle global calculé dans chaque sous-groupe. « global » = toutes les prédictions hors pli regroupées (diffère légèrement de la moyenne des plis). Détail par couple module × présentation et PR-AUC : `results_by_module.json`.

**Conception B, fenêtre 12 sem. (retraits futurs seulement) — par module**

| Modèle | AAA | BBB | CCC | DDD | EEE | FFF | GGG | global | étendue |
|---|---|---|---|---|---|---|---|---|---|
| *N (retraits)* | 693 (71) | 5943 (577) | 3045 (615) | 4818 (843) | 2377 (200) | 6130 (809) | 2288 (126) | 25294 (3241) | |
| Aggregated(8) | 0.5948 | 0.5596 | 0.5629 | 0.5579 | 0.6335 | 0.6281 | 0.5525 | 0.6059 | 0.0810 |
| Raw sequence | 0.5576 | 0.6012 | 0.6169 | 0.5926 | 0.6963 | 0.6518 | 0.5576 | 0.6466 | 0.1387 |
| AE embedding (64) | 0.5075 | 0.5824 | 0.6035 | 0.5906 | 0.6734 | 0.6405 | 0.5374 | 0.6377 | 0.1659 |
| Combined (AE + 8 desc.) | 0.5267 | 0.5854 | 0.6043 | 0.5860 | 0.6768 | 0.6444 | 0.5525 | 0.6403 | 0.1501 |

**Conception B, fenêtre 12 sem. (retraits futurs seulement) — par présentation**

| Modèle | 2013B | 2013J | 2014B | 2014J | global | étendue |
|---|---|---|---|---|---|---|
| *N (retraits)* | 3746 (478) | 7184 (834) | 5899 (820) | 8465 (1109) | 25294 (3241) | |
| Aggregated(8) | 0.6119 | 0.6029 | 0.5981 | 0.6125 | 0.6059 | 0.0144 |
| Raw sequence | 0.6441 | 0.6388 | 0.6499 | 0.6527 | 0.6466 | 0.0139 |
| AE embedding (64) | 0.6132 | 0.6398 | 0.6497 | 0.6369 | 0.6377 | 0.0365 |
| Combined (AE + 8 desc.) | 0.6122 | 0.6456 | 0.6525 | 0.6390 | 0.6403 | 0.0404 |

**Conception B, fenêtre 4 sem. — par module**

| Modèle | AAA | BBB | CCC | DDD | EEE | FFF | GGG | global | étendue |
|---|---|---|---|---|---|---|---|---|---|
| *N (retraits)* | 718 (96) | 6227 (861) | 3479 (1049) | 5227 (1252) | 2507 (330) | 6542 (1221) | 2341 (179) | 27041 (4988) | |
| Aggregated(8) | 0.4953 | 0.5144 | 0.5572 | 0.5420 | 0.6265 | 0.5763 | 0.5529 | 0.5798 | 0.1312 |
| Raw sequence | 0.4443 | 0.5479 | 0.5807 | 0.5951 | 0.6264 | 0.6008 | 0.5323 | 0.6206 | 0.1821 |
| AE embedding (64) | 0.4946 | 0.5425 | 0.5592 | 0.5872 | 0.6383 | 0.5950 | 0.5509 | 0.6161 | 0.1437 |
| Combined (AE + 8 desc.) | 0.4769 | 0.5342 | 0.5666 | 0.5924 | 0.6468 | 0.5926 | 0.5827 | 0.6179 | 0.1699 |

**Conception B, fenêtre 4 sem. — par présentation**

| Modèle | 2013B | 2013J | 2014B | 2014J | global | étendue |
|---|---|---|---|---|---|---|
| *N (retraits)* | 4026 (758) | 7535 (1185) | 6381 (1302) | 9099 (1743) | 27041 (4988) | |
| Aggregated(8) | 0.5582 | 0.5749 | 0.5720 | 0.5920 | 0.5798 | 0.0338 |
| Raw sequence | 0.5976 | 0.6238 | 0.6285 | 0.6196 | 0.6206 | 0.0309 |
| AE embedding (64) | 0.5974 | 0.6106 | 0.6201 | 0.6201 | 0.6161 | 0.0227 |
| Combined (AE + 8 desc.) | 0.5892 | 0.6209 | 0.6252 | 0.6182 | 0.6179 | 0.0360 |

**Conception A, cours entier (39 sem.) — par module**

| Modèle | AAA | BBB | CCC | DDD | EEE | FFF | GGG | global | étendue |
|---|---|---|---|---|---|---|---|---|---|
| *N (retraits)* | 735 (113) | 6622 (1255) | 3983 (1553) | 5745 (1767) | 2685 (508) | 7092 (1768) | 2366 (204) | 29228 (7168) | |
| Aggregated(8) | 0.9317 | 0.8927 | 0.9224 | 0.8964 | 0.9215 | 0.9140 | 0.8713 | 0.9099 | 0.0604 |
| Raw sequence | 0.9639 | 0.9127 | 0.9279 | 0.9144 | 0.9297 | 0.9241 | 0.9005 | 0.9267 | 0.0634 |
| Last active week (1 var.) | 0.9582 | 0.9097 | 0.9346 | 0.9087 | 0.9317 | 0.9187 | 0.8999 | 0.9168 | 0.0583 |
| AE embedding (64) | 0.9484 | 0.9080 | 0.9302 | 0.9050 | 0.9292 | 0.9200 | 0.8963 | 0.9231 | 0.0522 |
| Combined (AE + 8 desc.) | 0.9534 | 0.9103 | 0.9334 | 0.9070 | 0.9303 | 0.9208 | 0.8981 | 0.9248 | 0.0553 |

**Conception A, cours entier (39 sem.) — par présentation**

| Modèle | 2013B | 2013J | 2014B | 2014J | global | étendue |
|---|---|---|---|---|---|---|
| *N (retraits)* | 4261 (993) | 7973 (1616) | 6851 (1772) | 10143 (2787) | 29228 (7168) | |
| Aggregated(8) | 0.8937 | 0.9040 | 0.8885 | 0.9342 | 0.9099 | 0.0457 |
| Raw sequence | 0.9046 | 0.9240 | 0.9088 | 0.9454 | 0.9267 | 0.0408 |
| Last active week (1 var.) | 0.8986 | 0.9099 | 0.8944 | 0.9439 | 0.9168 | 0.0495 |
| AE embedding (64) | 0.8914 | 0.9194 | 0.9076 | 0.9440 | 0.9231 | 0.0526 |
| Combined (AE + 8 desc.) | 0.8974 | 0.9197 | 0.9078 | 0.9462 | 0.9248 | 0.0487 |

**Écart entre conceptions par module (séquence brute, AUC)**

| Module | A, 39 sem. | B, 12 sem. | B, 4 sem. |
|---|---|---|---|
| AAA | 0.9639 | 0.5576 | 0.4443 |
| BBB | 0.9127 | 0.6012 | 0.5479 |
| CCC | 0.9279 | 0.6169 | 0.5807 |
| DDD | 0.9144 | 0.5926 | 0.5951 |
| EEE | 0.9297 | 0.6963 | 0.6264 |
| FFF | 0.9241 | 0.6518 | 0.6008 |
| GGG | 0.9005 | 0.5576 | 0.5323 |

## 11. Sensibilité au découpage en plis (résultats actuels vs anciens résultats, `origin_folds/`)

Toutes les sections de ce document reposent sur une seule partition `GroupKFold(5)` (scikit-learn 1.9.1, déterministe). Les résultats antérieurs avaient été calculés sur une autre machine, dont la partition différait (ordre des groupes de même taille) et n'avait pas été conservée ; ils sont gardés dans `origin_folds/`. Mêmes données, mêmes représentations, mêmes hyperparamètres ; la partition change (la version de XGBoost de la machine d'origine n'a pas été enregistrée et peut aussi contribuer) :

| Conception | Comparaisons | Écart absolu moyen de l'AUC moyen | Écart absolu maximal | Cas le plus éloigné |
|---|---|---|---|---|
| B (retraits futurs) | 16 | 0.0035 | 0.0108 | Aggregated(8) [B1-like], 12 sem. (-0.0108) |
| A (tous les étudiants) | 20 | 0.0007 | 0.0045 | Aggregated(8), 4 sem. (+0.0045) |

Pour comparaison, l'écart-type entre plis d'un même modèle va de 0.0030 à 0.0161 en conception B (section 2).

## 12. Bootstrap apparié par étudiant (2000 rééchantillons ; AUC sur les prédictions hors pli regroupées)

Les étudiants (toutes leurs inscriptions ensemble) sont rééchantillonnés avec remise ; différence d'AUC-ROC entre deux modèles évalués sur le même rééchantillon ; IC95 % par percentiles. L'intervalle reflète l'incertitude d'échantillonnage des étudiants, les modèles étant fixés : il n'inclut pas la variabilité d'entraînement (graine : section 8 ; partition : section 11). Ces intervalles remplacent ceux de la section 4, qui ne reposent que sur 5 valeurs par modèle.

**Conception B (retraits futurs)**

| Comparaison | 4 sem. | 8 sem. | 12 sem. | 20 sem. |
|---|---|---|---|---|
| AE embedding (64) − C4 untrained LSTM encoder | +0.0044 [-0.0032, +0.0120] | +0.0189 [+0.0107, +0.0267] | +0.0101 [+0.0008, +0.0195] | -0.0067 [-0.0170, +0.0037] |
| AE embedding (64) − C3 PCA-64 | +0.0074 [+0.0003, +0.0149] | +0.0007 [-0.0072, +0.0087] | +0.0044 [-0.0047, +0.0137] | -0.0063 [-0.0172, +0.0041] |
| AE embedding (64) − Raw sequence | -0.0045 [-0.0120, +0.0027] | -0.0075 [-0.0155, +0.0003] | -0.0090 [-0.0180, +0.0005] | -0.0177 [-0.0284, -0.0075] |
| Combined (AE + 8 desc.) − Aggregated(8) | +0.0382 [+0.0296, +0.0471] | +0.0444 [+0.0350, +0.0538] | +0.0344 [+0.0239, +0.0450] | +0.0357 [+0.0235, +0.0474] |
| Combined (AE + 8 desc.) − Raw sequence | -0.0027 [-0.0100, +0.0045] | -0.0029 [-0.0111, +0.0048] | -0.0064 [-0.0155, +0.0028] | -0.0194 [-0.0294, -0.0098] |
| Raw sequence − Aggregated(8) | +0.0408 [+0.0319, +0.0502] | +0.0473 [+0.0377, +0.0571] | +0.0408 [+0.0304, +0.0517] | +0.0551 [+0.0429, +0.0671] |

**Conception B, 12 sem. — modèles supervisés (moyenne des graines)**

| Comparaison | Différence [IC95 %] | Part des rééchantillons > 0 |
|---|---|---|
| B3' pretrained − B3' scratch | +0.0037 [-0.0002, +0.0076] | 97.0% |
| B3' pretrained − B3 rerun | +0.0051 [+0.0014, +0.0091] | 99.5% |
| B3' scratch − B3 rerun | +0.0014 [-0.0010, +0.0041] | 87.1% |
| B3' pretrained − AE frozen + LogReg | +0.0048 [-0.0001, +0.0095] | 97.1% |
| B3' pretrained − AE frozen + XGBoost | +0.0091 [+0.0017, +0.0163] | 99.2% |
| B3' pretrained − Raw sequence | +0.0001 [-0.0072, +0.0072] | 51.1% |
| B3' scratch − Raw sequence | -0.0036 [-0.0107, +0.0038] | 17.0% |

**Conception A (tous les étudiants)**

| Comparaison | 4 sem. | 8 sem. | 12 sem. | 20 sem. | 39 sem. |
|---|---|---|---|---|---|
| Raw sequence − Aggregated(8) | +0.0332 [+0.0274, +0.0390] | +0.0332 [+0.0283, +0.0384] | +0.0276 [+0.0236, +0.0319] | +0.0327 [+0.0293, +0.0361] | +0.0168 [+0.0146, +0.0192] |
| Raw sequence − Last active week (1 var.) | +0.0822 [+0.0745, +0.0900] | +0.0584 [+0.0525, +0.0644] | +0.0376 [+0.0330, +0.0422] | +0.0293 [+0.0259, +0.0328] | +0.0099 [+0.0077, +0.0121] |
| Aggregated(8) − Last active week (1 var.) | +0.0490 [+0.0408, +0.0569] | +0.0252 [+0.0187, +0.0318] | +0.0100 [+0.0046, +0.0149] | -0.0034 [-0.0072, +0.0009] | -0.0069 [-0.0093, -0.0045] |

**Conception A, 39 sem. — embedding v11**

| Comparaison | Différence [IC95 %] |
|---|---|
| AE embedding (64) − Raw sequence | -0.0036 [-0.0056, -0.0018] |
| Combined (AE + 8 desc.) − Aggregated(8) | +0.0149 [+0.0128, +0.0173] |
| Combined (AE + 8 desc.) − Raw sequence | -0.0019 [-0.0037, -0.0001] |

**AUC par modèle, conception B, 12 sem. [IC95 % bootstrap]**

| Modèle | Graines | AUC [IC95 %] |
|---|---|---|
| Aggregated(8) | 1 | 0.6059 [0.5953, 0.6162] |
| Raw sequence | 1 | 0.6466 [0.6371, 0.6564] |
| AE embedding (64) | 1 | 0.6377 [0.6275, 0.6483] |
| Combined (AE + 8 desc.) | 1 | 0.6403 [0.6297, 0.6505] |
| C3 PCA-64 | 1 | 0.6333 [0.6236, 0.6433] |
| C4 untrained LSTM encoder | 1 | 0.6276 [0.6177, 0.6378] |
| B3' pretrained | 3 | 0.6468 [0.6372, 0.6554] |
| B3' scratch | 3 | 0.6431 [0.6334, 0.6521] |
| B3 rerun | 3 | 0.6416 [0.6320, 0.6507] |
| AE frozen + XGBoost | 1 | 0.6377 [0.6275, 0.6483] |
| AE frozen + LogReg | 1 | 0.6420 [0.6319, 0.6513] |

## 13. Bootstrap apparié par étudiant (2000 rééchantillons ; AUC calculée dans chaque pli puis moyennée sur les plis : statistique citée dans l'article, D-031)

Les étudiants (toutes leurs inscriptions ensemble) sont rééchantillonnés avec remise ; différence d'AUC-ROC entre deux modèles évalués sur le même rééchantillon ; IC95 % par percentiles. L'intervalle reflète l'incertitude d'échantillonnage des étudiants, les modèles étant fixés : il n'inclut pas la variabilité d'entraînement (graine : section 8 ; partition : section 11). Ces intervalles remplacent ceux de la section 4, qui ne reposent que sur 5 valeurs par modèle.

**Conception B (retraits futurs)**

| Comparaison | 4 sem. | 8 sem. | 12 sem. | 20 sem. |
|---|---|---|---|---|
| AE embedding (64) − C4 untrained LSTM encoder | +0.0044 [-0.0033, +0.0120] | +0.0187 [+0.0105, +0.0266] | +0.0103 [+0.0011, +0.0197] | -0.0069 [-0.0171, +0.0037] |
| AE embedding (64) − C3 PCA-64 | +0.0074 [+0.0004, +0.0149] | +0.0009 [-0.0070, +0.0088] | +0.0047 [-0.0045, +0.0140] | -0.0062 [-0.0170, +0.0043] |
| AE embedding (64) − Raw sequence | -0.0045 [-0.0120, +0.0027] | -0.0074 [-0.0154, +0.0003] | -0.0089 [-0.0180, +0.0004] | -0.0181 [-0.0287, -0.0080] |
| Combined (AE + 8 desc.) − Aggregated(8) | +0.0382 [+0.0296, +0.0472] | +0.0443 [+0.0349, +0.0541] | +0.0345 [+0.0238, +0.0450] | +0.0349 [+0.0227, +0.0467] |
| Combined (AE + 8 desc.) − Raw sequence | -0.0026 [-0.0100, +0.0045] | -0.0027 [-0.0108, +0.0051] | -0.0066 [-0.0158, +0.0026] | -0.0200 [-0.0300, -0.0102] |
| Raw sequence − Aggregated(8) | +0.0408 [+0.0318, +0.0502] | +0.0470 [+0.0374, +0.0569] | +0.0411 [+0.0307, +0.0521] | +0.0549 [+0.0426, +0.0669] |

**Conception B, 12 sem. — modèles supervisés (moyenne des graines)**

| Comparaison | Différence [IC95 %] | Part des rééchantillons > 0 |
|---|---|---|
| B3' pretrained − B3' scratch | +0.0059 [+0.0020, +0.0099] | 99.9% |
| B3' pretrained − B3 rerun | +0.0069 [+0.0033, +0.0108] | 100.0% |
| B3' scratch − B3 rerun | +0.0010 [-0.0015, +0.0037] | 78.1% |
| B3' pretrained − AE frozen + LogReg | +0.0091 [+0.0044, +0.0138] | 100.0% |
| B3' pretrained − AE frozen + XGBoost | +0.0133 [+0.0057, +0.0205] | 100.0% |
| B3' pretrained − Raw sequence | +0.0043 [-0.0030, +0.0113] | 87.4% |
| B3' scratch − Raw sequence | -0.0016 [-0.0091, +0.0057] | 33.6% |
| B3 rerun − Raw sequence | -0.0026 [-0.0101, +0.0046] | 23.4% |

**Conception A (tous les étudiants)**

| Comparaison | 4 sem. | 8 sem. | 12 sem. | 20 sem. | 39 sem. |
|---|---|---|---|---|---|
| Raw sequence − Aggregated(8) | +0.0330 [+0.0271, +0.0388] | +0.0331 [+0.0282, +0.0382] | +0.0276 [+0.0235, +0.0318] | +0.0327 [+0.0293, +0.0361] | +0.0168 [+0.0146, +0.0192] |
| Raw sequence − Last active week (1 var.) | +0.0768 [+0.0703, +0.0828] | +0.0553 [+0.0506, +0.0600] | +0.0354 [+0.0310, +0.0397] | +0.0275 [+0.0245, +0.0307] | +0.0094 [+0.0073, +0.0117] |
| Aggregated(8) − Last active week (1 var.) | +0.0438 [+0.0372, +0.0504] | +0.0222 [+0.0169, +0.0276] | +0.0078 [+0.0030, +0.0124] | -0.0052 [-0.0087, -0.0013] | -0.0074 [-0.0098, -0.0049] |

**Conception A, 39 sem. — embedding v11**

| Comparaison | Différence [IC95 %] |
|---|---|
| AE embedding (64) − Raw sequence | -0.0037 [-0.0056, -0.0018] |
| Combined (AE + 8 desc.) − Aggregated(8) | +0.0149 [+0.0127, +0.0172] |
| Combined (AE + 8 desc.) − Raw sequence | -0.0020 [-0.0038, -0.0001] |
| Combined + XGBoost − B3 supervised LSTM | +0.0014 [-0.0003, +0.0030] |
| Embedding + XGBoost − B3 supervised LSTM | -0.0004 [-0.0021, +0.0013] |
| Combined + XGBoost − Embedding + XGBoost | +0.0017 [+0.0008, +0.0027] |
| Embedding + XGBoost − Aggregated + XGBoost | +0.0132 [+0.0108, +0.0155] |
| Combined + XGBoost − Raw sequence | -0.0020 [-0.0038, -0.0001] |
| B3 supervised LSTM − Raw sequence | -0.0033 [-0.0051, -0.0015] |
| B3 supervised LSTM − Last active week (1 var.) | +0.0061 [+0.0043, +0.0082] |
| B2 XGBoost + demographics − B1 RandomForest | +0.0110 [+0.0092, +0.0127] |
| Combined + XGBoost − Last active week (1 var.) | +0.0075 [+0.0052, +0.0098] |

**AUC par modèle, conception B, 12 sem. [IC95 % bootstrap]**

| Modèle | Graines | AUC [IC95 %] |
|---|---|---|
| Aggregated(8) | 1 | 0.6060 [0.5952, 0.6163] |
| Raw sequence | 1 | 0.6471 [0.6377, 0.6568] |
| AE embedding (64) | 1 | 0.6381 [0.6281, 0.6486] |
| Combined (AE + 8 desc.) | 1 | 0.6405 [0.6301, 0.6508] |
| C3 PCA-64 | 1 | 0.6335 [0.6237, 0.6436] |
| C4 untrained LSTM encoder | 1 | 0.6278 [0.6179, 0.6380] |
| B3' pretrained | 3 | 0.6514 [0.6414, 0.6604] |
| B3' scratch | 3 | 0.6455 [0.6355, 0.6546] |
| B3 rerun | 3 | 0.6445 [0.6347, 0.6536] |
| AE frozen + XGBoost | 1 | 0.6381 [0.6281, 0.6486] |
| AE frozen + LogReg | 1 | 0.6423 [0.6321, 0.6515] |

## 14. Sensibilité à la définition de la cohorte (XGBoost, AUC-ROC moyen sur 5 plis)

**S1 — inscriptions sans aucune activité incluses** (3365 inscriptions ajoutées, dont 2988 retraits ; plis recalculés sur la cohorte étendue)

| Conception, fenêtre | N | dont ajoutées (retraits) | Modèle | Cohorte étendue | Cohorte d'analyse seule, mêmes prédictions |
|---|---|---|---|---|---|
| A, 39 sem. | 32593 | 3365 (2988) | Raw sequence | 0.9336 | 0.9254 |
| A, 39 sem. | 32593 | 3365 (2988) | Aggregated(8) | 0.9231 | 0.9098 |
| A, 39 sem. | 32593 | 3365 (2988) | Last active week (1 var.) | 0.9284 | 0.9176 |
| B, 12 sem. | 25733 | 439 (63) | Raw sequence | 0.6451 | 0.6466 |
| B, 12 sem. | 25733 | 439 (63) | Aggregated(8) | 0.6084 | 0.6095 |
| B, 12 sem. | 25733 | 439 (63) | Last active week (1 var.) | 0.5714 | 0.5714 |

**S2 — 74 retraits sans date de désinscription retirés des ensembles à risque** (conception B, plis de l'analyse principale)

| Fenêtre | Retraits futurs (analyse principale) | Modèle | Analyse principale | Sans ces retraits | Écart |
|---|---|---|---|---|---|
| 4 | 4988 | Raw sequence | 0.6207 | 0.6185 | -0.0023 |
| 4 | 4988 | Aggregated(8) | 0.5799 | 0.5776 | -0.0023 |
| 4 | 4988 | AE embedding (64) | 0.6162 | 0.6119 | -0.0044 |
| 8 | 4014 | Raw sequence | 0.6440 | 0.6378 | -0.0062 |
| 8 | 4014 | Aggregated(8) | 0.5970 | 0.5871 | -0.0099 |
| 8 | 4014 | AE embedding (64) | 0.6366 | 0.6349 | -0.0016 |
| 12 | 3241 | Raw sequence | 0.6471 | 0.6434 | -0.0037 |
| 12 | 3241 | Aggregated(8) | 0.6060 | 0.5991 | -0.0070 |
| 12 | 3241 | AE embedding (64) | 0.6381 | 0.6311 | -0.0071 |
| 20 | 1823 | Raw sequence | 0.7023 | 0.6962 | -0.0061 |
| 20 | 1823 | Aggregated(8) | 0.6474 | 0.6435 | -0.0040 |
| 20 | 1823 | AE embedding (64) | 0.6842 | 0.6730 | -0.0111 |

## 15. Pré-entraînement de B3′ selon la graine de l'autoencodeur (12 sem., 3 graines supervisées par autoencodeur ; bootstrap en moyenne des plis)

B3′ entraîné de zéro : 0.6455.

| Graine de l'autoencodeur | Perte de validation min. de l'autoencodeur | B3′ pré-entraîné (AUC moyen) | Gain sur B3′ de zéro [IC95 %] |
|---|---|---|---|
| 42 | 0.3079 | 0.6514 | +0.0059 [+0.0020, +0.0099] |
| 43 | 0.2905 | 0.6506 | +0.0051 [+0.0011, +0.0090] |
| 44 | 0.2783 | 0.6515 | +0.0060 [+0.0020, +0.0099] |
| 45 | 0.2711 | 0.6526 | +0.0071 [+0.0029, +0.0114] |
| 46 | 0.2536 | 0.6548 | +0.0093 [+0.0051, +0.0137] |
| toutes (15 modèles) | | 0.6522 | +0.0067 [+0.0030, +0.0106] |

B3′ pré-entraîné (toutes graines) − séquence brute + XGBoost : +0.0051 [-0.0018, +0.0119].

## 16. Comparaisons de l'autoencodeur en moyenne sur 5 graines (conception B, bootstrap en moyenne des plis)

| Comparaison | 4 sem. | 8 sem. | 12 sem. | 20 sem. |
|---|---|---|---|---|
| Embedding (5 seeds) − C4 untrained (5 seeds) | +0.0049 [+0.0003, +0.0095] | +0.0149 [+0.0093, +0.0201] | +0.0054 [-0.0005, +0.0118] | -0.0068 [-0.0136, +0.0002] |
| Embedding (5 seeds) − C3 PCA-64 | +0.0061 [+0.0004, +0.0121] | -0.0002 [-0.0068, +0.0064] | +0.0031 [-0.0044, +0.0107] | -0.0051 [-0.0137, +0.0037] |
| Embedding (5 seeds) − Raw sequence | -0.0059 [-0.0118, -0.0002] | -0.0085 [-0.0151, -0.0021] | -0.0105 [-0.0181, -0.0029] | -0.0171 [-0.0259, -0.0088] |
| Combined (5 seeds) − Aggregated(8) | +0.0381 [+0.0302, +0.0460] | +0.0407 [+0.0325, +0.0491] | +0.0325 [+0.0233, +0.0418] | +0.0416 [+0.0305, +0.0526] |
| Combined (5 seeds) − Raw sequence | -0.0027 [-0.0086, +0.0030] | -0.0063 [-0.0129, +0.0003] | -0.0086 [-0.0160, -0.0010] | -0.0133 [-0.0220, -0.0051] |

## 17. Contrôle « totaux par type d'activité » (C5) et « totaux hebdomadaires » (C2), conception B

| Modèle ou comparaison | 4 sem. | 8 sem. | 12 sem. | 20 sem. |
|---|---|---|---|---|
| AUC, C5 activity-type totals | 0.6165 | 0.6276 | 0.6229 | 0.6432 |
| AUC, C2 weekly totals | 0.5543 | 0.5929 | 0.5999 | 0.6769 |
| AUC, Raw sequence | 0.6207 | 0.6440 | 0.6471 | 0.7023 |
| AUC, Aggregated(8) | 0.5799 | 0.5970 | 0.6060 | 0.6474 |
| Raw sequence − C5 activity-type totals | +0.0043 [-0.0032, +0.0124] | +0.0164 [+0.0072, +0.0256] | +0.0242 [+0.0140, +0.0342] | +0.0591 [+0.0471, +0.0724] |
| C5 activity-type totals − Aggregated(8) | +0.0365 [+0.0278, +0.0449] | +0.0307 [+0.0209, +0.0406] | +0.0169 [+0.0062, +0.0281] | -0.0042 [-0.0168, +0.0078] |
| C5 activity-type totals − C2 weekly totals | +0.0622 [+0.0514, +0.0728] | +0.0347 [+0.0236, +0.0457] | +0.0230 [+0.0105, +0.0351] | -0.0337 [-0.0488, -0.0201] |
| Raw sequence − C2 weekly totals | +0.0665 [+0.0565, +0.0764] | +0.0511 [+0.0425, +0.0606] | +0.0471 [+0.0376, +0.0573] | +0.0254 [+0.0158, +0.0355] |
| C2 weekly totals − Aggregated(8) | -0.0257 [-0.0359, -0.0148] | -0.0040 [-0.0139, +0.0061] | -0.0061 [-0.0169, +0.0047] | +0.0295 [+0.0156, +0.0428] |

## 18. Graines × partitions : différences sur 5 partitions, embedding et encodeur non entraîné moyennés sur 5 graines (moyenne [min, max])

| Comparaison | 4 sem. | 8 sem. | 12 sem. | 20 sem. |
|---|---|---|---|---|
| Embedding − encodeur non entraîné | +0.0064 [+0.0035, +0.0078] | +0.0143 [+0.0106, +0.0168] | +0.0061 [+0.0032, +0.0078] | -0.0061 [-0.0083, -0.0033] |
| Embedding − PCA-64 | +0.0037 [+0.0021, +0.0054] | -0.0024 [-0.0060, +0.0033] | -0.0005 [-0.0061, +0.0049] | +0.0012 [-0.0018, +0.0071] |
| Embedding − séquence brute | -0.0051 [-0.0067, -0.0029] | -0.0053 [-0.0082, -0.0009] | -0.0090 [-0.0120, -0.0056] | -0.0162 [-0.0219, -0.0108] |

## 19. Pré-entraînement de B3′ : intervalle fondé sur les 5 plis (variance corrigée de Nadeau & Bengio)

| Graine de l'autoencodeur | Gain sur B3′ de zéro [IC95 %] |
|---|---|
| 42 | +0.0059 [+0.0003, +0.0115] |
| 43 | +0.0051 [-0.0071, +0.0172] |
| 44 | +0.0060 [-0.0041, +0.0162] |
| 45 | +0.0071 [-0.0060, +0.0201] |
| 46 | +0.0093 [-0.0013, +0.0199] |
| toutes | +0.0067 [-0.0031, +0.0164] |
