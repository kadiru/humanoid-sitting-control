"""Freeze exact graph elimination and independent reverse-row proof before execution."""
from low_seat_affine_output_578 import *
import datetime,shutil
def main():
 assert not (D/'affine_output_launch_578.json').exists() and not (D/'affine_output_preparation_578').exists()
 ce=read(D/'affine_output_compile_exit_578.json');assert ce['actual_exit']==0;checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_577.json','supplemental_preservation_577.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 names=['low_seat_affine_output_578.py','prepare_low_seat_affine_output_578.py','audit_low_seat_affine_output_579.py',Path(__file__).name,'low_seat_lifted_feasibility_490.py'];hashes={str(R/'scripts'/n):sha(R/'scripts'/n) for n in names}
 for folder in ['zero_face_preparation_572','zero_face_audit_573']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p;hashes[str(D/folder/p)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 for name in ['zero_face_launch_572.json','closure_record_577.json','zero_face_audit_exit_573.json','zero_face_preparation_exit_572.json']:
  p=D/name;hashes[str(p)]=sha(p)
 assert read(D/'zero_face_audit_573/independent_audit.json')['passed']
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),compile_actual_exit=0,compile_exit_evidence=ce,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,numerical_lp_calls=0,model_calls=0,physical_admission=False)
 (D/'affine_output_launch_578.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
