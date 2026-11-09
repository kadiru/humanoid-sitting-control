"""Freeze exact moment/gap dimension reduction and standalone reverse auditor before launch."""
from low_seat_balance_block_594 import *
import datetime,shutil

def main():
 assert not (D/'balance_block_launch_594.json').exists() and not (D/'balance_block_preparation_594').exists()
 ce=read(D/'balance_block_compile_exit_594.json');assert ce['actual_exit']==0;checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_593.json','supplemental_preservation_593.json','supplemental_authoring_preservation_593.json','supplemental_automation_verification_593.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 names=['low_seat_balance_block_594.py','prepare_low_seat_balance_block_594.py','freeze_low_seat_balance_block_594.py','audit_low_seat_balance_block_595.py']
 hashes={str(R/'scripts'/n):sha(R/'scripts'/n) for n in names}
 for folder in ['foot_block_preparation_586','foot_block_audit_587']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p;hashes[str(D/folder/p)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 for n in ['foot_block_launch_586.json','foot_block_preparation_exit_586.json','foot_block_audit_exit_587.json','closure_record_593.json','balance_block_compile_exit_594.json']:
  p=D/n;hashes[str(p)]=sha(p)
 assert read(D/'foot_block_audit_587/independent_audit.json')['passed']
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),compile_actual_exit=0,compile_exit_evidence=ce,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,physical_admission=False)
 (D/'balance_block_launch_594.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
