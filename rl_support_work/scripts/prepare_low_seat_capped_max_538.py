"""Prepare exact original epigraph split and NEW small proximal arithmetic ONCE."""
from low_seat_capped_max_538 import *
import datetime,shutil
def main():
 launch=read(D/'capped_max_launch_538.json');assert read(D/'capped_max_freeze_exit_538.json')['actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 out=D/'capped_max_preparation_538';out.mkdir(exist_ok=False);results={}
 for role in ['primary','witness']:
  src=D/'exact_row_quotient_preparation_516'/role/'original';base=D/'lifted_feasibility_preparation_490'/role;o=out/role;o.mkdir()
  A=sparse.load_npz(src/'matrix.npz').tocsr();arr=dict(np.load(src/'arrays.npz'));E=sparse.load_npz(base/'equality.npz')
  assert E.shape==(1366,A.shape[1]) and E.nnz==0 and not np.any(arr['equality_rhs'])
  ids=np.load(src.parent/'row_ids.npy');assert len(ids)==A.shape[0]
  results[role]=split(A,arr,ids,o);sparse.save_npz(o/'native_equality.npz',E[:,:-1].tocsr())
  results[role]['zero_equality_rows']=1366;results[role]['original_equality_source']=str(base/'equality.npz');results[role]['original_equality_sha256']=sha(base/'equality.npz')
  print(json.dumps(dict(role=role,**results[role])),flush=True)
 small=examples();(out/'small_exact_examples.json').write_text(json.dumps(small,indent=2))
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,specification=SPEC,systems=results,new_small_proximal_examples=6,new_small_fenchel_dual_lift_examples=3,lp_calls=0,full_size_proximal_updates=0,model_calls=0,root_calls=0,finite_maps=0,new_solver=False,new_candidate=False,physical_admission=False)
 (out/'preparation.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
