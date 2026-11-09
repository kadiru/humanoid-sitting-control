"""Freeze exact zero-face construction and independent audit before execution."""
from low_seat_zero_face_572 import *
import datetime,shutil
def main():
 assert not (D/'zero_face_launch_572.json').exists() and not (D/'zero_face_preparation_572').exists()
 ce=read(D/'zero_face_compile_exit_572.json');assert ce['actual_exit']==0;checks=[]
 for name in ['prior_hashes.json','coupled_support_evidence_preservation_571.json','coupled_support_supplemental_571.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 names=['low_seat_zero_face_572.py','prepare_low_seat_zero_face_572.py','audit_low_seat_zero_face_573.py',Path(__file__).name];hashes={str(R/'scripts'/n):sha(R/'scripts'/n) for n in names}
 for folder in ['coupled_support_preparation_570','coupled_support_audit_571','box_support_preparation_546','box_support_audit_547','exact_row_quotient_audit_517','definition_preparation_558','definition_audit_559']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p;hashes[str(D/folder/p)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 # Bind559's original546 inputs to the same source used by571, without replaying its pair identities.
 old=read(D/'definition_launch_558.json')['source_input_sha256']
 for p,h in old.items():
  if '/box_support_preparation_546/' in p:assert sha(p)==h,p;assert hashes[p]==h
 for name in ['definition_launch_558.json','closure_record_569.json','preparation_handoff_record_571.json','lifted_feasibility_preparation_490/primary/arrays.npz','lifted_feasibility_preparation_490/primary/equality.npz']:
  p=D/name;hashes[str(p)]=sha(p)
 for folder in ['coupled_support_audit_571','box_support_audit_547','exact_row_quotient_audit_517','definition_audit_559']:assert read(D/folder/'independent_audit.json')['passed']
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),compile_actual_exit=0,compile_exit_evidence=ce,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,numerical_lp_calls=0,model_calls=0,physical_admission=False)
 (D/'zero_face_launch_572.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
