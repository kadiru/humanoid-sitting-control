"""Independent saved-subset and separation-decision audit; no solver/model/map."""
from pathlib import Path
import numpy as np
from scipy import sparse
import json,hashlib,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'row_separation_audit_511';P=D/'row_separation_preparation_510';TOP=331495
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(A,B):
 x=(A-B).tocsr();return x.nnz==0 or np.all(x.data==0)
def main():
 assert read(D/'row_separation_preparation_exit_510.json')['actual_exit']==0
 for p,h in read(D/'row_separation_launch_510.json')['source_input_sha256'].items():assert sha(p)==h,p
 for p,h in read(P/'SHA256.json').items():assert sha(P/p)==h,p
 O.mkdir(exist_ok=False);details={};row=np.load(D/'clock_row_preparation_v2_446/row_arrays.npz');pathids=np.flatnonzero(row['keep'])
 assert len(pathids)==TOP
 for mode in ['primary','witness']:
  original=dict(np.load(D/'lifted_feasibility_preparation_490'/mode/'arrays.npz'));transformed=dict(np.load(D/'mixed_coordinate_preparation_504'/mode/'arrays.npz'));M=sparse.load_npz(D/'mixed_coordinate_preparation_504'/mode/'matrix.npz').tocsr()
  # Independent boolean selection, preserving native row order.
  select=np.ones(M.shape[0],dtype=bool);select[:TOP]=original['rhs'][:TOP]<=0;expected=np.arange(M.shape[0])[select];ids=np.load(P/mode/'row_ids.npy');assert np.array_equal(ids,expected)
  assert eq(sparse.load_npz(P/mode/'matrix.npz'),M[select]);saved=dict(np.load(P/mode/'arrays.npz'))
  assert np.array_equal(saved['rhs'],transformed['rhs'][select])
  for key in ['cost','bounds','equality_rhs']:assert np.array_equal(saved[key],transformed[key])
  assert np.array_equal(np.load(P/mode/'selected_final_path_row_ids.npy'),pathids[select[:TOP]]) and np.array_equal(np.load(P/mode/'deferred_final_path_row_ids.npy'),pathids[~select[:TOP]])
  # Closure+,closure-,closed+,closed-,and witness interior+/- are ALL present.
  boundaries=[TOP,TOP+68250,TOP+136500,TOP+136500+92267,TOP+136500+184534]
  if mode=='witness':boundaries += [boundaries[-1]+92267,boundaries[-1]+184534]
  assert boundaries[-1]==M.shape[0] and np.all(select[TOP:])
  details[mode]=dict(selected_rows=int(select.sum()),selected_path_rows=int(select[:TOP].sum()),deferred_path_rows=int((~select[:TOP]).sum()),master_nnz=int(M[select].nnz),full_nnz=int(M.nnz),coefficient_rhs_error=0,mandatory_block_boundaries=boundaries)
 cases=read(P/'separation_cases.json')
 for c in cases:
  selected=set(c['rows']);ro=c['original'];rt=c['transformed'];n=len(ro);good=c['finite'] and all(np.isfinite(v) for v in ro+rt+[c['other']]);active=max([0.,c['other']]+[ro[j] for j in selected]+[rt[j] for j in selected]) if good else None
  full=max([0.,c['other']]+ro+rt) if good else None
  cuts=[j for j in range(n) if j not in selected and (ro[j]>0 or rt[j]>0)] if good else []
  action='terminal'
  if c['status']==0 and good and active<=1e-7:action='add_all_violated_rows' if cuts else ('full_witness' if full<=1e-7 else 'terminal')
  ans=c['result'];assert ans['action']==action and ans['full_witness']==(action=='full_witness') and ans['active_violation']==active and ans['full_violation']==full
  assert ans['new_rows']==(cuts if action=='add_all_violated_rows' else [])
  assert ans['next_rows']==sorted(selected.union(cuts) if action=='add_all_violated_rows' else selected)
  assert ans['omitted_positive_original']==sum(j not in selected and ro[j]>0 for j in range(n)) and ans['omitted_positive_transformed']==sum(j not in selected and rt[j]>0 for j in range(n))
 assert len(cases)==10
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,assemblies=details,new_separation_cases_replayed=10,coefficient_rhs_error=0,smallest_positive_deferred_residual_not_waived=True,nonoptimal_and_active_failure_terminal=True,monotone_cut_union_verified=True,old_basis_inverse_incidence_fixtures_repeated=0,lp_calls=0,model_calls=0,finite_maps=0,new_actual_path=False,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({'independent_audit.json':sha(O/'independent_audit.json')},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
