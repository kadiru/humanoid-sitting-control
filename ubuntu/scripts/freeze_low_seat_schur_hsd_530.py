"""Freeze new HSD subsolver preparation, small test runs and independent audit."""
from low_seat_schur_hsd_530 import *
import datetime,shutil
R=Path('/home/kadir/humanoid-sitting')
def main():
 assert not (D/'schur_hsd_launch_530.json').exists() and not (D/'schur_hsd_preparation_530').exists();ce=read(D/'schur_hsd_compile_exit_530.json');assert ce['actual_exit']==0
 checks=[]
 for name in ['prior_hashes.json','arrow_operator_evidence_preservation_529.json','arrow_operator_supplemental_529.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 names=['low_seat_schur_hsd_530.py','prepare_low_seat_schur_hsd_530.py','audit_low_seat_schur_hsd_531.py',Path(__file__).name];hashes={str(R/'scripts'/n):sha(R/'scripts'/n) for n in names}
 for folder in ['torque_projection_preparation_522','torque_projection_audit_523','arrow_operator_preparation_528','arrow_operator_audit_529']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p;hashes[str(D/folder/p)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),compile_actual_exit=0,compile_exit_evidence=ce,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,new_full_size_LP_calls=0,new_small_LP_calls_authorized=2,physical_admission=False)
 (D/'schur_hsd_launch_530.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
