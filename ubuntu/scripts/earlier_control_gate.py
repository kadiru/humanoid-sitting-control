"""Frozen development-only progression and effort-retention criteria."""
from earlier_control_common import *
def gate(path,seed):
    rows=read(path/'results.json')['episodes'];assert len(rows)==18 and all(x['steps']==1000 for x in rows)
    parent=read(OLD/f'ppo_01/seed_{PARENTS[seed]}/development_final/results.json')['episodes']
    ref=read(OLD/'preflight/calibrate/results.json')['episodes']
    reasons=[]
    for x,y in zip(rows,parent):
        assert x['case']==y['case']
        if not x['success'] or any(x['failures'].values()):reasons.append(f"case{x['case']}: stability")
        for key in ['slip_ticks','illegal_ticks']:
            if x[key]:reasons.append(f"case{x['case']}: {key}")
        if x['traction_bad_ticks']>y['traction_bad_ticks']:reasons.append(f"case{x['case']}: added margin ticks")
    effort={}
    for window,tolerance in [('metrics_whole_trial',1.001),('metrics_last5s',1.01)]:
        avg=lambda a:sum(x[window]['total_squared_torque_Nm2'] for x in a)/len(a)
        value,p,r=avg(rows),avg(parent),avg(ref);effort[window]=dict(policy=value,parent=p,reference=r,parent_ratio=value/p,reference_ratio=value/r)
        if value>p*tolerance+1e-6:reasons.append(window+': parent effort bound')
        if value>=r:reasons.append(window+': reference savings lost')
    return dict(passed=not reasons,reasons=reasons,effort=effort,stable=sum(x['success'] for x in rows),strict=sum(x['strict_transfer'] for x in rows))
