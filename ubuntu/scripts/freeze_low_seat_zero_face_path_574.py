"""Freeze ONE complete full-size574 method and all575 independent branches; no old calculation replay."""
from low_seat_zero_face_execution_574 import SPEC
from low_seat_zero_face_input_574 import D,sha,read
from pathlib import Path
import json,datetime,shutil
R=Path('/home/kadir/humanoid-sitting')
def main():
 assert not (D/'zero_face_path_launch_574.json').exists() and not (D/'zero_face_path_574').exists();ce=read(D/'zero_face_path_compile_exit_574.json');assert ce['actual_exit']==0;checks=[]
 for name in ['prior_hashes.json','zero_face_evidence_preservation_573.json','zero_face_supplemental_573.json']:
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
 names=['low_seat_zero_face_input_574.py','low_seat_zero_face_execution_574.py','solve_low_seat_zero_face_lp_574.py','build_low_seat_zero_face_path_574.py','freeze_low_seat_zero_face_path_574.py','certify_low_seat_zero_face_input_575.py','audit_low_seat_zero_face_solver_575.py','audit_low_seat_zero_face_steps_575.py','audit_low_seat_zero_face_path_575.py','audit_low_seat_zero_face_result_575.py']
 # New-input branch retains full original rows; no box/quotient producer used.
 for n in names:hashes[str(R/'scripts'/n)]=sha(R/'scripts'/n)
 for folder in ['box_support_preparation_546','box_support_audit_547','coupled_support_preparation_570','coupled_support_audit_571','zero_face_preparation_572','zero_face_audit_573','definition_audit_559','exact_row_quotient_audit_517']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p;hashes[str(D/folder/p)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 exits={n:read(D/n) for n in ['box_support_preparation_exit_546.json','box_support_audit_exit_547.json','torque_path_exit_524.json','torque_audit_exit_525.json','coupled_support_preparation_exit_570.json','coupled_support_audit_exit_571.json','zero_face_preparation_exit_572.json','zero_face_audit_exit_573.json']};assert all(v['actual_exit']==0 for v in exits.values())
 for p,h in read(D/'zero_face_launch_572.json')['source_input_sha256'].items():
  assert sha(p)==h,p;hashes[p]=h
 for n in ['zero_face_launch_572.json','preparation_handoff_record_573.json','zero_face_preservation_exit_573.json']:
  p=D/n
  if p.exists():hashes[str(p)]=sha(p)
 import scipy
 runtime=dict(scipy_version=scipy.__version__,python_version=__import__('sys').version,backend='scipy linprog highs-ipm, cold, no old basis')
 opt=Path(scipy.__file__).parent/'optimize'
 for p in [opt/'_linprog.py',opt/'_linprog_highs.py']+list((opt/'_highspy').glob('*.py'))+list((opt/'_highspy').glob('*.so')):hashes[str(p)]=sha(p)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),compile_actual_exit=0,compile_exit_evidence=ce,dependencies_actual_exit_verified=True,dependency_exits=exits,prior_checks=checks,workers=workers,source_input_sha256=hashes,specification=SPEC,installed_backend=runtime,disk_free_bytes=shutil.disk_usage(D).free,initial_assembly_repeated=False,old_metric_or_small_tests_repeated=False,no_torque_projection=True,physical_admission=False)
 (D/'zero_face_path_launch_574.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['source_input_sha256','specification']},indent=2),flush=True)
if __name__=='__main__':main()
