"""Freeze new rational-to-binary64 input representation and independent audit before execution."""
from low_seat_graph_encoding_580 import *
import datetime
def main():
 assert not (D/'graph_encoding_launch_580.json').exists() and not (D/'graph_encoding_preparation_580').exists()
 ce=read(D/'graph_encoding_compile_exit_580.json');assert ce['actual_exit']==0;checks=[]
 for name in ['prior_hashes.json','affine_output_evidence_preservation_579.json','affine_output_supplemental_579.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 names=['low_seat_graph_encoding_580.py','prepare_low_seat_graph_encoding_580.py','audit_low_seat_graph_encoding_581.py',Path(__file__).name]
 hashes={str(R/'scripts'/n):sha(R/'scripts'/n) for n in names}
 for folder in ['affine_output_preparation_578','affine_output_audit_579','box_support_preparation_546','box_support_audit_547']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p;hashes[str(D/folder/p)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 for name in ['affine_output_launch_578.json','box_support_launch_546.json','closure_record_577.json','preparation_handoff_record_579.json','affine_output_audit_exit_579.json','affine_output_preparation_exit_578.json','box_support_audit_exit_547.json']:
  p=D/name;hashes[str(p)]=sha(p)
 for name in ['affine_output_audit_579','box_support_audit_547']:assert read(D/name/'independent_audit.json')['passed']
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),compile_actual_exit=0,compile_exit_evidence=ce,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,numerical_lp_calls=0,model_calls=0,physical_admission=False)
 (D/'graph_encoding_launch_580.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
