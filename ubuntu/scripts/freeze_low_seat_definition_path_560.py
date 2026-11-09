"""Freeze ONE complete full-size560 method and all561 independent branches; no old calculation replay."""
from low_seat_definition_execution_560 import SPEC
from low_seat_definition_input_560 import D,sha,read
from pathlib import Path
import json,datetime,shutil
R=Path('/home/kadir/humanoid-sitting')
def main():
 assert not (D/'definition_path_launch_560.json').exists() and not (D/'definition_path_560').exists();ce=read(D/'definition_path_compile_exit_560.json');assert ce['actual_exit']==0;checks=[]
 for name in ['prior_hashes.json','definition_evidence_preservation_559.json','definition_supplemental_559.json']:
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
 names=['low_seat_definition_input_560.py','low_seat_definition_execution_560.py','solve_low_seat_definition_lp_560.py','build_low_seat_definition_path_560.py','freeze_low_seat_definition_path_560.py','certify_low_seat_definition_input_561.py','audit_low_seat_definition_solver_561.py','audit_low_seat_definition_steps_561.py','audit_low_seat_definition_path_561.py','audit_low_seat_definition_result_561.py']
 names+=['low_seat_box_input_548.py','certify_low_seat_box_input_549.py']
 for n in names:hashes[str(R/'scripts'/n)]=sha(R/'scripts'/n)
 for folder in ['box_support_preparation_546','box_support_audit_547','definition_preparation_558','definition_audit_559']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p;hashes[str(D/folder/p)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 exits={n:read(D/n) for n in ['box_support_preparation_exit_546.json','box_support_audit_exit_547.json','torque_path_exit_524.json','torque_audit_exit_525.json','definition_preparation_exit_558.json','definition_audit_exit_559.json']};assert all(v['actual_exit']==0 for v in exits.values())
 import scipy
 runtime=dict(scipy_version=scipy.__version__,python_version=__import__('sys').version,backend='scipy linprog highs-ipm, cold, no old basis')
 opt=Path(scipy.__file__).parent/'optimize'
 for p in [opt/'_linprog.py',opt/'_linprog_highs.py']+list((opt/'_highspy').glob('*.py'))+list((opt/'_highspy').glob('*.so')):hashes[str(p)]=sha(p)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),compile_actual_exit=0,compile_exit_evidence=ce,dependencies_actual_exit_verified=True,dependency_exits=exits,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,installed_backend=runtime,disk_free_bytes=shutil.disk_usage(D).free,initial_assembly_repeated=False,old_metric_or_small_tests_repeated=False,no_torque_projection=True,physical_admission=False)
 (D/'definition_path_launch_560.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
