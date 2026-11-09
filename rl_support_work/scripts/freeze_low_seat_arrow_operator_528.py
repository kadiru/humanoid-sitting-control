"""Freeze the new operator factorization preparation and independent audit."""
from low_seat_arrow_operator_528 import *
import datetime,shutil
R=Path('/home/kadir/humanoid-sitting')
def main():
 assert not (D/'arrow_operator_launch_528.json').exists() and not (D/'arrow_operator_preparation_528').exists()
 ce=read(D/'arrow_operator_compile_exit_528.json');assert ce['actual_exit']==0;checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_527.json','supplemental_preservation_527.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 names=['low_seat_arrow_operator_528.py','prepare_low_seat_arrow_operator_528.py','audit_low_seat_arrow_operator_529.py',Path(__file__).name]
 hashes={str(R/'scripts'/n):sha(R/'scripts'/n) for n in names}
 for folder in ['torque_projection_preparation_522','torque_projection_audit_523']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p;hashes[str(D/folder/p)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 assert read(D/'torque_projection_audit_523/independent_audit.json')['passed']
 p=D/'lifted_feasibility_preparation_490/reference.npz';hashes[str(p)]=sha(p)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),compile_actual_exit=0,compile_exit_evidence=ce,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,lp_calls=0,model_calls=0,finite_maps=0,physical_admission=False)
 (D/'arrow_operator_launch_528.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
