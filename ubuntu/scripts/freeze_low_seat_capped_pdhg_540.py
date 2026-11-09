"""Freeze the NEW capped-maximum PDHG subsolver preparation and independent audit."""
from low_seat_capped_pdhg_540 import *
import datetime,shutil
R=Path('/home/kadir/humanoid-sitting')
def main():
 assert not (D/'capped_pdhg_launch_540.json').exists() and not (D/'capped_pdhg_preparation_540').exists()
 ce=read(D/'capped_pdhg_compile_exit_540.json');assert ce['actual_exit']==0;checks=[]
 for name in ['prior_hashes.json','capped_max_evidence_preservation_539.json','capped_max_supplemental_539.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 names=['low_seat_capped_pdhg_540.py','prepare_low_seat_capped_pdhg_540.py','audit_low_seat_capped_pdhg_541.py',Path(__file__).name];hashes={str(R/'scripts'/n):sha(R/'scripts'/n) for n in names}
 for folder in ['capped_max_preparation_538','capped_max_audit_539']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p;hashes[str(D/folder/p)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 assert read(D/'capped_max_audit_539/independent_audit.json')['passed']
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),compile_actual_exit=0,compile_exit_evidence=ce,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,new_full_size_LP_calls=0,new_small_LP_calls_planned=2,physical_admission=False)
 (D/'capped_pdhg_launch_540.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
