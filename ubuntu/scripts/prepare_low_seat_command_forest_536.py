"""Build new exact forest multirow implications once, using saved systems only."""
from low_seat_command_forest_536 import *
import datetime,shutil
def main():
 launch=read(D/'command_forest_launch_536.json');assert read(D/'command_forest_freeze_exit_536.json')['actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 out=D/'command_forest_preparation_536';out.mkdir(exist_ok=False)
 extra=np.load(D/'lifted_feasibility_preparation_490/reference.npz')['extra_rows'];results={}
 for role in ['primary','witness']:
  src=D/'torque_projection_preparation_522'/role;o=out/role;o.mkdir();A=sparse.load_npz(src/'matrix.npz').tocsr();arr=dict(np.load(src/'arrays.npz'));ret=np.load(src/'projection_ledger.npz')['retained_columns']
  results[role]=construct(A,arr,ret,extra,o);shutil.copy2(src/'equality.npz',o/'equality.npz');print(json.dumps(dict(role=role,**results[role])),flush=True)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,specification=SPEC,systems=results,lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,physical_admission=False)
 (out/'preparation.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
