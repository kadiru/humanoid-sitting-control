"""Freeze ONE complete full-size542 method and all543 independent branches; no old calculation replay."""
from low_seat_capped_execution_542 import SPEC
from low_seat_capped_solver_542 import D,sha,read
from pathlib import Path
import json,datetime,shutil
R=Path('/home/kadir/humanoid-sitting')
def main():
 assert not (D/'capped_path_launch_542.json').exists() and not (D/'capped_path_542').exists();ce=read(D/'capped_path_compile_exit_542.json');assert ce['actual_exit']==0;checks=[]
 for name in ['prior_hashes.json','capped_pdhg_evidence_preservation_541.json','capped_pdhg_supplemental_541.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 # Bind all immutable model/finite/initial and actual-audit dependencies already frozen532.
 hashes=read(D/'schur_path_launch_532.json')['source_input_sha256'].copy()
 for p,h in hashes.items():assert sha(p)==h,p
 names=['low_seat_capped_solver_542.py','low_seat_capped_execution_542.py','solve_low_seat_capped_lp_542.py','build_low_seat_capped_path_542.py','freeze_low_seat_capped_path_542.py','certify_low_seat_capped_input_543.py','audit_low_seat_capped_solver_543.py','audit_low_seat_capped_steps_543.py','audit_low_seat_capped_path_543.py','audit_low_seat_capped_result_543.py','low_seat_capped_max_538.py','low_seat_capped_pdhg_540.py','audit_low_seat_capped_pdhg_541.py']
 for n in names:hashes[str(R/'scripts'/n)]=sha(R/'scripts'/n)
 for folder in ['capped_max_preparation_538','capped_max_audit_539','capped_pdhg_preparation_540','capped_pdhg_audit_541']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p;hashes[str(D/folder/p)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 exits={n:read(D/n) for n in ['capped_max_preparation_exit_538.json','capped_max_audit_exit_539.json','capped_pdhg_preparation_exit_540.json','capped_pdhg_audit_exit_541.json','torque_path_exit_524.json','torque_audit_exit_525.json']};assert all(v['actual_exit']==0 for v in exits.values())
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),compile_actual_exit=0,compile_exit_evidence=ce,dependencies_actual_exit_verified=True,dependency_exits=exits,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,initial_assembly_repeated=False,old_metric_or_small_tests_repeated=False,no_torque_projection=True,physical_admission=False)
 (D/'capped_path_launch_542.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
