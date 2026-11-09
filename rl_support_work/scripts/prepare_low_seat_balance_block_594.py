"""One bounded exact saved moment/gap reduction; no numerical solver or primalpoint."""
import time,resource
START=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
from low_seat_balance_block_594 import *
import datetime

def main():
 assert read(D/'balance_block_freeze_exit_594.json')['actual_exit']==0
 for p,h in read(D/'balance_block_launch_594.json')['source_input_sha256'].items():assert sha(p)==h,p
 check(START);out=D/'balance_block_preparation_594';out.mkdir(exist_ok=False);progress={}
 try:q=build(out,START,progress)
 except BaseException as e:
  failure=dict(passed=False,exception=type(e).__name__,message=str(e),elapsed_s=time.monotonic()-START,progress=progress,partial_files_are_not_certificates=True,terminal=True,restart_allowed=False,physical_admission=False)
  (out/'terminal_failure.json').write_text(json.dumps(failure,indent=2));print(json.dumps(failure,indent=2),flush=True);raise
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,preparation_only=True,specification=SPEC,primary=q,elapsed_s=time.monotonic()-START,numerical_lp_calls=0,numerical_linear_system_solves=0,exact_block_coefficient_solves=1365,exact_block_inverses=1365,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,primal_vectors_constructed=0,analytic_dual_vectors_constructed=0,physical_admission=False,new_solver=False,new_candidate=False,peak_memory_measured=False)
 (out/'preparation.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
