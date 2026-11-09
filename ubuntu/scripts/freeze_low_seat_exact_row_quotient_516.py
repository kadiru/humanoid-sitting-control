"""Freeze one new exact-row quotient preparation and independent proof auditor."""
from low_seat_exact_row_quotient_516 import *
import datetime,shutil
R=Path('/home/kadir/humanoid-sitting')
def main():
 assert not (D/'exact_row_quotient_launch_516.json').exists() and not (D/'exact_row_quotient_preparation_516').exists()
 compile_exit=read(D/'exact_row_quotient_compile_exit_516.json');assert compile_exit['actual_exit']==0
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_515.json','supplemental_preservation_515.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 sources=['low_seat_exact_row_quotient_516.py','prepare_low_seat_exact_row_quotient_516.py','audit_low_seat_exact_row_quotient_517.py',Path(__file__).name]
 hashes={str(R/'scripts'/n):sha(R/'scripts'/n) for n in sources}
 for folder in ['lifted_feasibility_preparation_490','mixed_coordinate_preparation_504']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
  for mode in ['primary','witness']:
   for n in ['matrix.npz','arrays.npz','equality.npz']:hashes[str(D/folder/mode/n)]=sha(D/folder/mode/n)
 for folder in ['lifted_feasibility_audit_491','mixed_coordinate_audit_505']:
  p=D/folder/'independent_audit.json';assert read(p)['passed'];hashes[str(p)]=sha(p)
 record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,compile_actual_exit=0,compile_exit_evidence=compile_exit,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,no_rejected_or_partial_vector_input=True,lp_calls=0,model_calls=0,finite_maps=0,physical_admission=False)
 (D/'exact_row_quotient_launch_516.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
