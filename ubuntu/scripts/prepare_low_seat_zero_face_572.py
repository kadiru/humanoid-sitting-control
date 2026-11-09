"""One exact primary zero-face preparation and analytic dual construction."""
import time,resource
START=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
from low_seat_zero_face_572 import *
import datetime
def main():
 assert read(D/'zero_face_freeze_exit_572.json')['actual_exit']==0
 for p,h in read(D/'zero_face_launch_572.json')['source_input_sha256'].items():assert sha(p)==h,p
 check(START);out=D/'zero_face_preparation_572';out.mkdir(exist_ok=False);s=build(out,START)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,preparation_only=True,specification=SPEC,primary=s,elapsed_s=time.monotonic()-START,numerical_lp_calls=0,linear_system_solves=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,primal_vectors_constructed=0,analytic_lower_bound_dual_vectors=1,full_primal_dual_KKT_evaluations=0,physical_admission=False,new_solver=False,new_candidate=False)
 (out/'preparation.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
