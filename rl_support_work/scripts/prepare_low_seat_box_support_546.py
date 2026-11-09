"""One NEW exact box-support reduction preparation; no LP or old proof replay."""
from low_seat_box_support_546 import *
import datetime,resource,shutil
def main():
 start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
 assert read(D/'box_support_freeze_exit_546.json')['actual_exit']==0
 for p,h in read(D/'box_support_launch_546.json')['source_input_sha256'].items():assert sha(p)==h,p
 out=D/'box_support_preparation_546';out.mkdir(exist_ok=False);results={}
 for role in ['primary','witness']:
  src=D/'exact_row_quotient_preparation_516'/role/'original';o=out/role;o.mkdir()
  A=sparse.load_npz(src/'matrix.npz').tocsr();arr=dict(np.load(src/'arrays.npz'));ids=np.load(src.parent/'row_ids.npy')
  ep=D/'lifted_feasibility_preparation_490'/role/'equality.npz';E=sparse.load_npz(ep)
  assert E.shape==(1366,A.shape[1]) and E.nnz==0 and not np.any(arr['equality_rhs'])
  assert len(ids)==A.shape[0] and A.shape[1]==78791
  results[role]=derive(A,arr,ids,o,start);shutil.copy2(ep,o/'equality.npz')
  print(json.dumps(dict(role=role,**{k:v for k,v in results[role].items() if 'slack' not in k})),flush=True)
 check_time(start)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,preparation_only=True,specification=SPEC,systems=results,elapsed_s=time.monotonic()-start,lp_calls=0,small_examples=0,linear_solves=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,new_solver=False,new_candidate=False,physical_admission=False)
 (out/'preparation.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2))
 print(json.dumps(dict(passed=True,elapsed_s=r['elapsed_s'],removed=sum(s['proved_removed_rows'] for s in results.values()))),flush=True)
if __name__=='__main__':main()
