"""Bootstrap apparié par étudiant : AUC pondérée exacte (ex æquo compris) pour NB rééchantillons d'étudiants.
Même calcul et même tirage (graine, ordre) que la classe définie dans eval_bootstrap.py (T-10), isolés ici pour être
importables. `groups` = identifiant d'étudiant de chaque ligne ; `yy` = cible binaire."""
import numpy as np
CHUNK = 100
class Boot:
    def __init__(self, groups, yy, nb, seed=42):
        self.y = np.asarray(yy); _, self.g = np.unique(groups, return_inverse=True); ng = self.g.max() + 1
        rng = np.random.default_rng(seed)
        self.counts = [rng.multinomial(ng, np.full(ng, 1/ng), size=min(CHUNK, nb - b)).astype('float32') for b in range(0, nb, CHUNK)]
    def auc(self, s):
        """Vecteur des NB AUC pour un vecteur de scores."""
        u, inv = np.unique(s, return_inverse=True); order = np.argsort(inv, kind='stable'); starts = np.searchsorted(inv[order], np.arange(len(u)))
        out = []
        for C in self.counts:
            Wt = C[:, self.g]
            P = np.add.reduceat((Wt * self.y)[:, order], starts, axis=1); Ng = np.add.reduceat((Wt * (1 - self.y))[:, order], starts, axis=1)
            out.append((P * (np.cumsum(Ng, axis=1) - Ng + 0.5 * Ng)).sum(1) / (P.sum(1) * Ng.sum(1)))
        return np.concatenate(out)
    def model(self, seeds): return np.mean([self.auc(s) for s in seeds], axis=0)
    # --- statistique « moyenne des plis » : l'AUC est calculée DANS chaque pli de test puis moyennée sur les plis.
    # Contrairement à l'AUC sur prédictions regroupées, elle ne dépend pas de l'échelle des scores d'un pli à l'autre
    # (échelle qui varie pour les réseaux de neurones, d'où un désavantage artificiel face à XGBoost : D-031).
    def auc_folds(self, s, fold):
        folds = np.unique(fold); tot = 0
        for k in folds:
            mk = (fold == k); sk = s[mk]; yk = self.y[mk]; gk = self.g[mk]
            u, inv = np.unique(sk, return_inverse=True); order = np.argsort(inv, kind='stable'); starts = np.searchsorted(inv[order], np.arange(len(u)))
            out = []
            for C in self.counts:
                Wt = C[:, gk]
                P = np.add.reduceat((Wt * yk)[:, order], starts, axis=1); Ng = np.add.reduceat((Wt * (1 - yk))[:, order], starts, axis=1)
                out.append((P * (np.cumsum(Ng, axis=1) - Ng + 0.5 * Ng)).sum(1) / (P.sum(1) * Ng.sum(1)))
            tot = tot + np.concatenate(out)
        return tot / len(folds)
    def model_folds(self, seeds, fold): return np.mean([self.auc_folds(s, fold) for s in seeds], axis=0)
