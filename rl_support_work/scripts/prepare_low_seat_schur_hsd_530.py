"""Prepare exact dyadic source representations and run two new SMALL LP fixtures once."""
from low_seat_schur_hsd_530 import *
import datetime
O=D/'schur_hsd_preparation_530'
def main():
 launch=read(D/'schur_hsd_launch_530.json');assert read(D/'schur_hsd_freeze_exit_530.json')['actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 O.mkdir(exist_ok=False);systems={};tests={}
 for role in ['primary','witness']:
  src=D/'torque_projection_preparation_522'/role;out=O/role;out.mkdir();A=sparse.load_npz(src/'matrix.npz');arr=dict(np.load(src/'arrays.npz'));E=sparse.load_npz(src/'equality.npz');p=np.load(D/'arrow_operator_preparation_528'/role/'partition.npz')
  v=normalize(A,arr['rhs'],arr['cost'],arr['bounds'],E,arr['equality_rhs'])
  assert all(j in set(p['separator']) for j in v['bound_owners'][:,0]);sparse.save_npz(out/'matrix.npz',v['matrix']);sparse.save_npz(out/'cone_original.npz',v['original'])
  np.savez_compressed(out/'arrays.npz',rhs=v['rhs'],cost=v['cost'],original_rhs=v['original_rhs'],original_cost=v['original_cost'],row_power=v['row_power'],column_power=v['column_power'],bound_owners=v['bound_owners'],local=p['local'],separator=p['separator'],block_ptr=p['block_ptr'])
  systems[role]=dict(shape=list(v['matrix'].shape),nnz=v['matrix'].nnz,added_bound_rows=len(v['bound_owners']),local_columns=len(p['local']),separator_columns=len(p['separator']),new_full_size_LP_calls=0,new_full_size_local_factors=0,reference_witness_only=role=='witness')
 for test in fixtures():
  out=O/test['name'];out.mkdir();np.savez_compressed(out/'input.npz',**{k:v for k,v in test.items() if k!='name'})
  tests[test['name']]=solve_lp(sparse.csr_matrix(test['A']),test['b'],test['c'],test['bounds'],sparse.csr_matrix((0,3)),np.zeros(0),np.array([0,1]),np.array([2]),np.array([0,1,2]),out)
  print(json.dumps(dict(test=test['name'],result=tests[test['name']])),flush=True)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=all(r['status']==0 for r in tests.values()),specification=SPEC,systems=systems,small_LP_tests=tests,new_small_LP_calls=2,new_full_size_LP_calls=0,new_full_size_local_factors=0,old528_partitions_factors_examples_repeated=False,model_calls=0,root_calls=0,finite_maps=0,new_actual_path=False,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
