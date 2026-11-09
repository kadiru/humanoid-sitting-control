"""Freeze coupled mixed-rate/box proof policy and independent audit before execution."""
from low_seat_coupled_support_570 import *
import datetime,shutil
def main():
 assert not (D/'coupled_support_launch_570.json').exists() and not (D/'coupled_support_preparation_570').exists()
 ce=read(D/'coupled_support_compile_exit_570.json');assert ce['actual_exit']==0;checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_569.json','supplemental_preservation_569.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 names=['low_seat_coupled_support_570.py','prepare_low_seat_coupled_support_570.py','audit_low_seat_coupled_support_571.py',Path(__file__).name]
 hashes={str(R/'scripts'/n):sha(R/'scripts'/n) for n in names}
 for folder in ['box_support_preparation_546','box_support_audit_547','exact_row_quotient_audit_517']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p;hashes[str(D/folder/p)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 for role in ['primary','witness']:
  for f in ['matrix.npz','arrays.npz']:
   p=D/'exact_row_quotient_preparation_516'/role/'original'/f;hashes[str(p)]=sha(p)
 for name in ['clock_row_preparation_v2_446/row_arrays.npz','lifted_feasibility_preparation_490/reference.npz','closure_record_569.json']:
  p=D/name;hashes[str(p)]=sha(p)
 assert read(D/'exact_row_quotient_audit_517/independent_audit.json')['passed'] and read(D/'box_support_audit_547/independent_audit.json')['passed']
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),compile_actual_exit=0,compile_exit_evidence=ce,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,authoring_history='Read-only schema inspection e77eb4 exited1 after requesting nonexistent key group; printed schema showed groups. No scientific program or proof was run by that inspection. Source uses the observed groups key.',numerical_lp_calls=0,model_calls=0,physical_admission=False)
 (D/'coupled_support_launch_570.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
