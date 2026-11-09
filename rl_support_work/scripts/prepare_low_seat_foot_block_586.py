"""One bounded exact coupled foot-block dimension reduction, no numerical LP."""
import time,resource
START=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
from low_seat_foot_block_586 import *
import datetime
def main():
 assert read(D/'foot_block_freeze_exit_586.json')['actual_exit']==0
 for p,h in read(D/'foot_block_launch_586.json')['source_input_sha256'].items():assert sha(p)==h,p
 check(START);out=D/'foot_block_preparation_586';out.mkdir(exist_ok=False);q=build(out,START)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,preparation_only=True,specification=SPEC,primary=q,elapsed_s=time.monotonic()-START,numerical_lp_calls=0,numerical_linear_system_solves=0,exact_block_coefficient_solves=2730,exact_block_inverses=2730,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,primal_vectors_constructed=0,analytic_dual_vectors_constructed=0,physical_admission=False,new_solver=False,new_candidate=False)
 (out/'preparation.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
