"""Freeze NEW exact torque-auxiliary elimination preparation and independent audit."""
from low_seat_torque_projection_522 import *
import datetime,shutil
R=Path('/home/kadir/humanoid-sitting')
def main():
 assert not (D/'torque_projection_launch_522.json').exists() and not (D/'torque_projection_preparation_522').exists();compile_exit=read(D/'torque_projection_compile_exit_522.json');assert compile_exit['actual_exit']==0
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_521.json','supplemental_preservation_521.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 names=['low_seat_torque_projection_522.py','prepare_low_seat_torque_projection_522.py','audit_low_seat_torque_projection_523.py',Path(__file__).name];hashes={str(R/'scripts'/n):sha(R/'scripts'/n) for n in names}
 for folder in ['exact_row_quotient_preparation_516','exact_row_quotient_audit_517']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p;hashes[str(D/folder/p)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 assert read(D/'exact_row_quotient_audit_517/independent_audit.json')['passed'];assert read(D/'exact_row_quotient_preparation_516/preparation.json')['passed'] is False
 for p in [D/'lifted_feasibility_preparation_490/reference.npz']+[D/'lifted_feasibility_preparation_490'/role/'equality.npz' for role in ['primary','witness']]:hashes[str(p)]=sha(p)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,compile_actual_exit=0,compile_exit_evidence=compile_exit,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,failed516_fixture_excluded=True,no_partial_or_rejected_vector_input=True,lp_calls=0,model_calls=0,finite_maps=0,physical_admission=False)
 (D/'torque_projection_launch_522.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
