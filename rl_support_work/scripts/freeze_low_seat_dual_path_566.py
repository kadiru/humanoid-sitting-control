"""Freeze ONE complete full-size566 method and all567 independent branches; no old calculation replay."""
from low_seat_dual_execution_566 import SPEC
from low_seat_dual_input_566 import D,sha,read
from pathlib import Path
import json,datetime,shutil
R=Path('/home/kadir/humanoid-sitting')
def main():
 assert not (D/'dual_path_launch_566.json').exists() and not (D/'dual_path_566').exists();ce=read(D/'dual_path_compile_exit_566.json');assert ce['actual_exit']==0;checks=[]
 for name in ['prior_hashes.json','dual_balance_evidence_preservation_565.json','dual_balance_supplemental_565.json']:
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
 names=['low_seat_dual_input_566.py','low_seat_dual_execution_566.py','solve_low_seat_dual_lp_566.py','build_low_seat_dual_path_566.py','freeze_low_seat_dual_path_566.py','certify_low_seat_dual_input_567.py','audit_low_seat_dual_solver_567.py','audit_low_seat_dual_steps_567.py','audit_low_seat_dual_path_567.py','audit_low_seat_dual_result_567.py']
 names+=['low_seat_box_input_548.py','certify_low_seat_box_input_549.py']
 for n in names:hashes[str(R/'scripts'/n)]=sha(R/'scripts'/n)
 for folder in ['box_support_preparation_546','box_support_audit_547','dual_balance_preparation_564','dual_balance_audit_565']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p;hashes[str(D/folder/p)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 exits={n:read(D/n) for n in ['box_support_preparation_exit_546.json','box_support_audit_exit_547.json','torque_path_exit_524.json','torque_audit_exit_525.json','dual_balance_preparation_exit_564.json','dual_balance_audit_exit_565.json']};assert all(v['actual_exit']==0 for v in exits.values())
 for n in ['dual_balance_audit_original_exit_565.json','dual_balance_audit_stage_565.json','dual_balance_audit_565.log']:
  p=D/n
  if p.exists():hashes[str(p)]=sha(p)
 import scipy
 runtime=dict(scipy_version=scipy.__version__,python_version=__import__('sys').version,backend='scipy linprog highs-ipm, cold, no old basis')
 opt=Path(scipy.__file__).parent/'optimize'
 for p in [opt/'_linprog.py',opt/'_linprog_highs.py']+list((opt/'_highspy').glob('*.py'))+list((opt/'_highspy').glob('*.so')):hashes[str(p)]=sha(p)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),compile_actual_exit=0,compile_exit_evidence=ce,dependencies_actual_exit_verified=True,dependency_exits=exits,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,installed_backend=runtime,disk_free_bytes=shutil.disk_usage(D).free,initial_assembly_repeated=False,old_metric_or_small_tests_repeated=False,no_torque_projection=True,physical_admission=False)
 (D/'dual_path_launch_566.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
