"""Saved-only scope index for the completed whole-row separation run."""
from pathlib import Path
import json,datetime
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def main():
 assert read(D/'separated_path_exit_512.json')['actual_exit']==0
 P=D/'separated_path_512';result=read(P/'results.json');history=[json.loads(s) for s in (P/'history.jsonl').read_text().splitlines()];stages=[];masters=[]
 for h in history:
  for role in ['primary','interior']:
   if role not in h:continue
   s=h[role];stages.append(dict(major=h['major'],role=role,actual_lp_calls=s['actual_lp_calls'],master_assemblies=s['master_assemblies'],full_candidate=s['candidate'],terminal=s['terminal_failure'],reason=s['reason']))
   for i,r in enumerate(s['rounds']):
    R=Path(r['folder']);dec=read(R/'decision.json');w=r['worker'];masters.append(dict(major=h['major'],role=role,round=i,**w,decision=dec['action'],active_violation=dec.get('active_violation'),full_violation=dec.get('full_violation'),omitted_positive_original=dec.get('omitted_positive_original'),omitted_positive_transformed=dec.get('omitted_positive_transformed'),added_rows=len(dec.get('new_rows',[])),original_reconstructed=(R/'original_witness.npy').exists(),full_witness=dec['full_witness']))
 row=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'));mapping=np.flatnonzero(row['keep']);cuts=[]
 for R in sorted(P.glob('major_*/*/round_*')):
  if not (R/'full_residuals.npz').exists():continue
  z=dict(np.load(R/'full_residuals.npz'));dec=read(R/'decision.json');ids=np.asarray(dec.get('new_rows',[]),dtype=int);fullids=mapping[ids]
  groups=[]
  for name in dict.fromkeys(row['groups'][fullids].tolist()):
   sel=ids[row['groups'][fullids]==name];fi=mapping[sel];groups.append(dict(group=str(name),count=len(sel),hard_rows=int((~row['soft'][fi]).sum()),soft_rows=int(row['soft'][fi].sum()),maximum_original_LP_violation=float(z['original'][sel].max()),maximum_transformed_LP_violation=float(z['transformed'][sel].max())))
  cuts.append(dict(folder=str(R),new_rows=len(ids),same_positive_row_ids=bool(np.array_equal(np.flatnonzero(z['original']>0),np.flatnonzero(z['transformed']>0))),groups=groups,scope='Signed linear LP violations under the unexecuted partial-master vector, not actual nonlinear or physical residuals. Guard and epigraph remain included.'))
 actual=dict(np.load(P/'final.npz'));initial=dict(np.load(D/'free_gap_path_472/initial.npz'));canonical=np.load(D/'lifted_feasibility_preparation_490/reference.npz')['features']
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,cut_scope=cuts,stages=stages,masters=masters,actual_scipy_linprog_calls=sum(x['actual_lp_call'] for x in masters),returned_master_candidates=sum(x['candidate'] for x in masters),fully_certified_witnesses=result['full_linear_witnesses'],original_reconstructions=result['original_direction_reconstructions'],separation_cut_rounds=result['separation_cut_rounds'],separation_added_rows=result['separation_added_rows'],accepted_solver_iterates=result['accepted_solver_iterates'],worker_actual_exits=[x['worker_actual_exit'] for x in masters],worker_wall_cap_hits=sum(x['worker_wall_timeout'] for x in masters),actual_X_exact472=bool(np.array_equal(actual['X'],initial['X'])),actual_Y_exact472=bool(np.array_equal(actual['Y'],initial['Y'])),actual_J_exact472=bool(np.array_equal(actual['J'],initial['J'])),auxiliary_exact490canonical=bool(np.array_equal(actual['features'],canonical)),auxiliary_Y_equals_actual=bool(np.array_equal(actual['features'][:,24:],actual['Y'])),new_model_calls_in_this_index=0,new_matrix_calculations=0,new_lp_calls=0,new_finite_maps=0,physical_admission=False)
 assert out['actual_scipy_linprog_calls']==result['actual_lp_calls']
 (D/'separated_saved_scope_515.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
