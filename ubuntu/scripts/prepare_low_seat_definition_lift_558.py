"""ONE exact local-mechanics definition lift from saved source pairs; no numerical solver."""
from low_seat_definition_lift_558 import *
import resource,datetime
def main():
 start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3));assert read(D/'definition_freeze_exit_558.json')['actual_exit']==0
 for p,h in read(D/'definition_launch_558.json')['source_input_sha256'].items():assert sha(p)==h,p
 out=D/'definition_preparation_558';out.mkdir(exist_ok=False);systems={}
 for role in ['primary','witness']:
  o=out/role;o.mkdir();systems[role]=derive(role,o,start);print(json.dumps(dict(role=role,**systems[role])),flush=True)
 check(start);r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,preparation_only=True,specification=SPEC,systems=systems,elapsed_s=time.monotonic()-start,lp_calls=0,small_examples=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,new_solver=False,new_candidate=False,physical_admission=False,old_calculations_repeated=False)
 (out/'preparation.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(dict(passed=True,elapsed_s=r['elapsed_s'])),flush=True)
if __name__=='__main__':main()
