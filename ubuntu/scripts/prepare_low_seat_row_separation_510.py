"""Saved-row separation preparation only. No LP or complete run wrapper."""
from low_seat_row_separation_510 import *
import datetime
O=D/'row_separation_preparation_510'
def main():
 launch=read(D/'row_separation_launch_510.json');assert launch['compile_actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 O.mkdir(exist_ok=False);results={};old=D/'lifted_feasibility_preparation_490';mixed=D/'mixed_coordinate_preparation_504'
 for mode in ['primary','witness']:
  p=O/mode;p.mkdir();a=dict(np.load(old/mode/'arrays.npz'));t=dict(np.load(mixed/mode/'arrays.npz'));M=sparse.load_npz(mixed/mode/'matrix.npz').tocsr()
  ids=seed_rows(a['rhs']);validate_rows(ids,len(a['rhs']));np.save(p/'row_ids.npy',ids)
  sparse.save_npz(p/'matrix.npz',M[ids]);np.savez_compressed(p/'arrays.npz',rhs=t['rhs'][ids],cost=t['cost'],bounds=t['bounds'],equality_rhs=t['equality_rhs'])
  # Equality matrix is inherited by hash, not reassembled or relaxed.
  eqpath=mixed/mode/'equality.npz'
  mapping=np.flatnonzero(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz')['keep'])
  np.save(p/'selected_final_path_row_ids.npy',mapping[ids[ids<TOP]])
  np.save(p/'deferred_final_path_row_ids.npy',mapping[np.setdiff1d(np.arange(TOP),ids[ids<TOP])])
  results[mode]=dict(full_shape=list(M.shape),selected_rows=len(ids),deferred_path_rows=TOP-int((ids<TOP).sum()),selected_path_rows=int((ids<TOP).sum()),mandatory_rows=int((ids>=TOP).sum()),master_nnz=M[ids].nnz,full_nnz=M.nnz,all_consistency_signs_retained=True,all_intervals_retained=True,equality_source=str(eqpath),equality_sha256=sha(eqpath),optimistic_witness_fixture_only=mode=='witness')
 cases=fixture_cases();(O/'separation_cases.json').write_text(json.dumps(cases,indent=2))
 assert [c['result']['action'] for c in cases]==['add_all_violated_rows','add_all_violated_rows','terminal','terminal','add_all_violated_rows','full_witness','full_witness','terminal','add_all_violated_rows','terminal']
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,specification=SPEC,assemblies=results,new_separation_cases=len(cases),full_basis_rebuilt=False,model_calls=0,lp_calls=0,finite_maps=0,actual_X_Y_J_changed=False,new_complete_path=False,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
