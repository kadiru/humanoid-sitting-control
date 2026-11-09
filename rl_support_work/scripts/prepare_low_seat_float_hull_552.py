"""ONE new binary64 interval-hull preparation from saved546; no numerical witness algorithm."""
from low_seat_float_hull_552 import *
import resource,datetime
def main():
 start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3));assert read(D/'float_hull_freeze_exit_552.json')['actual_exit']==0
 for p,h in read(D/'float_hull_launch_552.json')['source_input_sha256'].items():assert sha(p)==h,p
 assert read(D/'box_support_audit_547/independent_audit.json')['passed']
 out=D/'float_hull_preparation_552';out.mkdir(exist_ok=False);systems={}
 for role in ['primary','witness']:
  o=out/role;o.mkdir();systems[role]=derive(role,o,start);print(json.dumps(dict(role=role,**systems[role])),flush=True)
 check(start);r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,preparation_only=True,specification=SPEC,systems=systems,elapsed_s=time.monotonic()-start,lp_calls=0,small_examples=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,new_solver=False,new_candidate=False,physical_admission=False,old_calculations_repeated=False)
 (out/'preparation.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(dict(passed=True,elapsed_s=r['elapsed_s'])),flush=True)
if __name__=='__main__':main()
