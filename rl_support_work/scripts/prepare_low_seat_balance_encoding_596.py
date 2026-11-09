"""One frozen numeric-input preparation for594; no numerical optimizer or primalpoint."""
import time,resource
START=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
from low_seat_balance_encoding_596 import *
import datetime

def main():
 out=D/'balance_encoding_preparation_596';progress={}
 assert read(D/'balance_encoding_freeze_exit_596.json')['actual_exit']==0
 assert not out.exists();out.mkdir()
 try:
  for p,h in read(D/'balance_encoding_launch_596.json')['source_input_sha256'].items():assert sha(p,START)==h,p
  check(START);q=build(out,START,progress)
  r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,preparation_only=True,specification=SPEC,primary=q,scientific_elapsed_s=time.monotonic()-START,numerical_lp_calls=0,numerical_linear_system_solves=0,exact_block_inverses=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,primal_vectors_constructed=0,analytic_dual_vectors_constructed=0,physical_admission=False,new_solver=False,new_candidate=False,scientific_execution_recovery=False,peak_memory_measured=False)
  (out/'preparation.json').write_text(json.dumps(r,indent=2));check(START)
  (out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p,START) for p in out.rglob('*') if p.is_file()},indent=2));check(START)
  timing=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),through_output_manifest_elapsed_s=time.monotonic()-START,limit_s=240,checked_limit_passed=True,excludes='Final tiny completion_timing.json write/stdout/processshutdown; not fullyinstrumentedprocesswall.')
  (out/'completion_timing.json').write_text(json.dumps(timing,indent=2));check(START);print(json.dumps(dict(result=r,completion=timing),indent=2),flush=True)
 except BaseException as e:
  failure=dict(passed=False,exception=type(e).__name__,message=str(e),elapsed_s=time.monotonic()-START,progress=progress,partial_prefix_is_not_certificate=True,terminal=True,restart_allowed=False,physical_admission=False)
  (out/'terminal_failure.json').write_text(json.dumps(failure,indent=2));print(json.dumps(failure,indent=2),flush=True);raise
if __name__=='__main__':main()
