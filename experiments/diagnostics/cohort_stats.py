"""Statistiques descriptives de la cohorte citées dans l'article (aucun chiffre saisi à la main dans le texte).
Lecture seule sur interim/. Écrit results/diagnostics/cohort_stats.json. Idempotent (FORCE=1 pour recalculer)."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'early_windows'))
from common import *
OUTJ = str(ROOT/'results'/'diagnostics'/'cohort_stats.json'); skip_if_exists(OUTJ, 'statistiques de cohorte')
keys = ['code_module', 'code_presentation', 'id_student']
st = pd.read_parquet(f'{UP}/student_table.parquet')
sv = pd.read_parquet(f'{UP}/student_vle.parquet', columns=keys + ['date', 'sum_click'])
act = [l for l in open(f'{UP}/activity_types.txt').read().split('\n') if l.strip()]
active = set(map(tuple, sv[keys].drop_duplicates().astype({'code_module': str, 'code_presentation': str}).values))
inact = ~st[keys].astype({'code_module': str, 'code_presentation': str}).apply(tuple, axis=1).isin(active)
ex = st[inact]
assert len(st) - len(ex) == len(cohort), "la cohorte reconstruite ne correspond pas à interim/cohort.parquet"
vc = cohort['id_student'].value_counts()
mod = cohort.groupby(cohort['code_module'].astype(str))['dropout'].agg(['size', 'mean'])
o = {
    'n_enrollments': int(len(st)),
    'n_excluded_no_vle': int(len(ex)), 'share_excluded': float(len(ex)/len(st)),
    'excluded_share_withdrawn': float((ex['final_result'] == 'Withdrawn').mean()),
    'excluded_median_unregistration_day': float(ex['date_unregistration'].astype('float').median()),
    'n_cohort': int(len(cohort)), 'dropout_rate': float(y.mean()), 'n_dropout': int(y.sum()),
    'n_unique_students': int(cohort['id_student'].nunique()), 'n_students_multiple_enrollments': int((vc > 1).sum()),
    'n_modules': int(cohort['code_module'].astype(str).nunique()), 'n_presentations': int(cohort['code_presentation'].astype(str).nunique()),
    'n_module_presentation_pairs': int(cohort.groupby([cohort['code_module'].astype(str), cohort['code_presentation'].astype(str)]).ngroups),
    'n_click_rows': int(len(sv)), 'share_click_rows_before_start': float((sv['date'] < 0).mean()),
    'n_activity_types': int(len(act)), 'sequence_shape': [int(v) for v in seq.shape],
    'dropout_rate_by_module': {k: {'n': int(r['size']), 'rate': float(r['mean'])} for k, r in mod.iterrows()},
    'withdrawn_without_unregistration_date': int(((y == 1) & np.isnan(unreg)).sum()),
    'non_withdrawn_with_unregistration_date': int(((y == 0) & ~np.isnan(unreg)).sum()),
    'median_unregistration_day_withdrawn': float(np.nanmedian(unreg[y == 1])),
}
json.dump(o, open(OUTJ, 'w'), indent=2); print(json.dumps(o, indent=2))
