import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
if all(os.path.exists(f'{ART}/agg_w{w}.parquet') for w in [4,8,12,20,39]) and not FORCE:
    print('[SKIP] tous les agg_w*.parquet existent (FORCE=1 pour recalculer)'); sys.exit(0)
import pyarrow.parquet as pq, gc
t = pq.read_table(f'{UP}/student_vle.parquet', columns=['code_module','code_presentation','id_student','id_site','date','sum_click'])
mod = t.column('code_module').to_pandas(); pres = t.column('code_presentation').to_pandas()
mp_cat = pd.Categorical((mod.astype(str)+'_'+pres.astype(str)))
cats = list(mp_cat.categories); code = mp_cat.codes.astype('int64')
ids = t.column('id_student').to_numpy().astype('int64')
site = t.column('id_site').to_numpy().astype('int32'); date = t.column('date').to_numpy().astype('int16'); clk = t.column('sum_click').to_numpy().astype('float32')
del t, mod, pres, mp_cat; gc.collect()
cmap = {c:i for i,c in enumerate(cats)}
ckey = (cohort['id_student'].values.astype('int64')*64 + (cohort['code_module'].astype(str)+'_'+cohort['code_presentation'].astype(str)).map(cmap).values.astype('int64'))
order = np.argsort(ckey); sk = ckey[order]
key = ids*64 + code; del ids, code
pos = np.searchsorted(sk, key); pos[pos>=len(sk)] = 0
ok = sk[pos]==key
row = np.where(ok, order[pos], -1).astype('int32'); del key,pos,ok
keep = row>=0
row, site, date, clk = row[keep], site[keep], date[keep], clk[keep]; gc.collect()
vle = pd.read_parquet(f'{UP}/vle.parquet')[['id_site','activity_type']]
amap = vle.set_index('id_site')['activity_type'].isin(['quiz','externalquiz','questionnaire'])
assess = pd.Series(site).map(amap).fillna(False).values.astype(bool)
week = np.clip(date//7, 0, 38).astype('int16')
print("lignes:", len(row), flush=True)
N = len(cohort)
for w in [4,8,12,20,39]:
    if os.path.exists(f'{ART}/agg_w{w}.parquet') and not FORCE:
        print(f'[SKIP] agg_w{w} existe', flush=True); continue
    m = date < 7*w
    r, s_, d_, c_, wk_, a_ = row[m], site[m], date[m], clk[m], week[m], assess[m]
    f = pd.DataFrame(index=np.arange(N))
    f['total_clicks'] = pd.Series(c_).groupby(r).sum()
    f['assessment_clicks'] = pd.Series(c_*a_).groupby(r).sum()
    f['active_days'] = pd.DataFrame({'r':r,'d':d_}).drop_duplicates().groupby('r').size()
    f['resource_diversity'] = pd.DataFrame({'r':r,'s':s_}).drop_duplicates().groupby('r').size()
    f['active_weeks'] = pd.DataFrame({'r':r,'w':wk_}).drop_duplicates().groupby('r').size()
    ws = pd.DataFrame({'r':r,'w':wk_,'c':c_}).groupby(['r','w'])['c'].sum().reset_index().groupby('r')['c'].agg(['max','mean','std'])
    f['peak_week_clicks']=ws['max']; f['avg_weekly_clicks']=ws['mean']; f['click_std']=ws['std']
    f = f[['total_clicks','active_weeks','active_days','resource_diversity','assessment_clicks','peak_week_clicks','avg_weekly_clicks','click_std']].fillna(0).astype('float32')
    f.to_parquet(f'{ART}/agg_w{w}.parquet'); print(w, f.shape, flush=True); del m,r,s_,d_,c_,wk_,a_; gc.collect()
