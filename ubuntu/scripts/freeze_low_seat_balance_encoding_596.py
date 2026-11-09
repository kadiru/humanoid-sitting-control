"""Freeze NEW594 numeric input/domain/exactE policy and independent597 before execution."""
from low_seat_balance_encoding_596 import *
import datetime

def main():
 assert not (D/'balance_encoding_launch_596.json').exists() and not (D/'balance_encoding_preparation_596').exists()
 ce=read(D/'balance_encoding_compile_exit_596.json');assert ce['actual_exit']==0;checks=[]
 for name in ['prior_hashes.json','balance_block_evidence_preservation_595.json','balance_block_supplemental_595.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 names=['low_seat_balance_encoding_596.py','prepare_low_seat_balance_encoding_596.py','freeze_low_seat_balance_encoding_596.py','audit_low_seat_balance_encoding_597.py']
 hashes={str(R/'scripts'/n):sha(R/'scripts'/n) for n in names}
 for folder in ['balance_block_preparation_594','balance_block_audit_595','foot_encoding_preparation_588','foot_encoding_audit_589']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p;hashes[str(D/folder/p)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 for n in ['balance_block_launch_594.json','balance_block_preparation_exit_594.json','balance_block_audit_exit_595.json','foot_encoding_launch_588.json','foot_encoding_preparation_exit_588.json','foot_encoding_audit_exit_589.json','preparation_handoff_record_595.json','closure_record_593.json','balance_encoding_compile_exit_596.json']:
  p=D/n;hashes[str(p)]=sha(p)
 assert read(D/'balance_block_audit_595/independent_audit.json')['passed'] and read(D/'foot_encoding_audit_589/independent_audit.json')['passed']
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),compile_actual_exit=0,compile_exit_evidence=ce,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,physical_admission=False)
 (D/'balance_encoding_launch_596.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
