"""Saved-array localization only; no model, root, derivative, solver or audit run."""
from pathlib import Path
import json,numpy as np,hashlib,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def main():
    r=json.loads((D/'direct_path_418/results.json').read_text());f=dict(np.load(D/'direct_path_418/final.npz'));e=dict(np.load(D/'direct_path_418/event_clock.npz'));w=dict(np.load(D/'direct_path_418/window.npz'));raw=e['raw_ineq'];iq=f['ineq'];hard=f['hard']&f['keep'];groups=[]
    def add(name,ids,unit):
        ids=np.array(list(ids));a=raw[ids];b=iq[ids];j=int(np.argmin(a));k=int(np.argmin(b));groups.append(dict(name=name,rows=len(ids),unit=unit,minimum_original=float(a[j]),minimum_original_index=int(ids[j]),negative_original_count=int((a<0).sum()),minimum_construction=float(b[k]),minimum_construction_index=int(ids[k]),negative_construction_count=int((b<0).sum())))
    for name,start,stop,unit in [('command_lower',0,16,'rad'),('command_upper',16,32,'rad'),('effort_upper',32,48,'Nm'),('effort_lower',48,64,'Nm'),('corner_nonnegative',64,72,'BW fraction'),('corner_upper',72,80,'BW fraction')]:add(name,[i*80+j for i in range(114) for j in range(start,stop)],unit)
    for name,cols,unit in [('root_lower',[0,1],'m'),('joint_lower',range(2,14),'rad'),('root_upper',[14,15],'m'),('joint_upper',range(16,28),'rad')]:add(name,[9120+i*28+j for i in range(103) for j in cols],unit)
    for g in json.loads((D/'support_preparation_371/preparation.json').read_text())['groups'][2:]:add(g['name'],range(g['start'],g['stop']),g['raw_unit'])
    assert sum(g['rows'] for g in groups)==20284
    negatives=[]
    for i in np.flatnonzero(hard&(iq<0)):
        assert i<20284;g=next(g for g in groups if False) if False else None
        negatives.append(dict(index=int(i),construction=float(iq[i]),original=float(raw[i]),original_failed=bool(raw[i]<0)))
    blocks=[]
    for label,arr in [('clock',e['clock_raw']),('window',w['raw'])]:
        names=['increasing_rate','decreasing_rate','upper_target','lower_target'] if label=='clock' else ['upper_target','lower_target']
        for j,name in enumerate(names):
            a=arr[:,j*16:(j+1)*16];k,col=map(int,np.unravel_index(a.argmin(),a.shape));blocks.append(dict(source=label,group=name,rows=a.size,minimum_original=float(a[k,col]),negative_count=int((a<0).sum()),sample=k,joint_column=col,unit='rad/tick' if 'rate' in name else 'rad'))
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,groups=groups,clock_window_groups=blocks,negative_hard_construction_rows=negatives,minimum_original_hard_row=float(raw[hard[:20284]].min()),minimum_construction_hard_row=float(iq[hard].min()),minimum_coordinate_bound=r['algorithm']['minimum_coordinate_bound'],solver_objective_evaluations_nfev=r['optimizer']['evaluations'],solver_jacobian_evaluations_njev=r['optimizer']['jacobian_evaluations'],legacy_actual_constraint_evaluations_field_is_nfev=True,distinct_constraint_or_root_evaluation_count=None,accounting_note='Source418 copied objective nfev into actual_constraint_evaluations. Original evidence retained; distinct constraint/root evaluations were not instrumented and remain unavailable. No physical/policy ticks are inferred.',physical_admission=False,physics_ticks=0,policy_optimizer_updates=0,source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),D/'direct_path_418/results.json',D/'direct_path_418/final.npz',D/'direct_path_418/event_clock.npz',D/'direct_path_418/window.npz']})
    target=D/'direct_saved_constraint_groups_421.json';assert not target.exists();target.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['groups','source_sha256']},indent=2))
if __name__=='__main__':main()
