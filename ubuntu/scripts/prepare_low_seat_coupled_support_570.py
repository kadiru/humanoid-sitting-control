"""One coupled-rate implication preparation; no LP or old proof replay."""
import time,resource
T0=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
from low_seat_coupled_support_570 import *
import datetime
def main():
 assert read(D/'coupled_support_freeze_exit_570.json')['actual_exit']==0
 for p,h in read(D/'coupled_support_launch_570.json')['source_input_sha256'].items():assert sha(p)==h,p
 check(T0);out=D/'coupled_support_preparation_570';out.mkdir(exist_ok=False);systems={}
 for role in ['primary','witness']:
  o=out/role;o.mkdir();systems[role]=derive(role,o,T0);print(json.dumps(dict(role=role,**systems[role])),flush=True)
 check(T0);result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,preparation_only=True,specification=SPEC,systems=systems,elapsed_s=time.monotonic()-T0,numerical_lp_calls=0,robot_lp_calls=0,linear_system_solves=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,old_scientific_calculations_repeated=False,physical_admission=False)
 (out/'preparation.json').write_text(json.dumps(result,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(dict(passed=True,elapsed_s=result['elapsed_s'])),flush=True)
if __name__=='__main__':main()
