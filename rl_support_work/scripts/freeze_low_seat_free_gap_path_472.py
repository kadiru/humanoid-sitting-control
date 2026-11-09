"""Freeze ONE new executable free-gap run and its independent audit sources."""
from pathlib import Path
import json,datetime,shutil
from low_seat_free_gap_execution_472 import D,sha,SPEC
R=Path('/home/kadir/humanoid-sitting');read=lambda p:json.loads(Path(p).read_text())
def main():
 assert not (D/'free_gap_path_launch_472.json').exists() and not (D/'free_gap_path_472').exists()
 ex=read(D/'free_gap_preparation_exits_471.json')
 for key in ['chart468_actual_exit','staged_audit469v2_actual_exit','assembly470_actual_exit','audit471_actual_exit']:assert ex[key]==0
 compile_exit=read(D/'free_gap_compile_exit_472.json');assert compile_exit['actual_exit']==0
 for folder in ['exact_gap_chart_audit_v2_469','free_gap_assembly_audit_471']:assert read(D/folder/'independent_audit.json')['passed']
 checks=[]
 for name in ['prior_hashes.json','free_gap_evidence_preservation_471.json','free_gap_supplemental_471.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 sources=['low_seat_free_gap_execution_472.py','build_low_seat_free_gap_path_472.py','audit_low_seat_free_gap_steps_473.py','audit_low_seat_free_gap_path_473.py','freeze_low_seat_free_gap_path_472.py','low_seat_free_gap_solver_470.py','low_seat_exact_gap_chart_468.py','low_seat_gap_coordinates_v2_467.py','low_seat_simplex_retraction_460.py','audit_low_seat_simplex_v2_461.py','low_seat_collocation_solver_448.py','low_seat_clock_transcription_444.py']
 hashes={str(R/'scripts'/name):sha(R/'scripts'/name) for name in sources}
 for folder in ['exact_gap_chart_preparation_468','exact_gap_chart_audit_v2_469','free_gap_solver_preparation_470','free_gap_assembly_audit_471','gap_coordinate_preparation_v3_466','contact_lift_440','contact_lift_audit_441','clock_transcription_preparation_444']:
  for p,h in read(D/folder/'SHA256.json').items():
   q=D/folder/p;assert sha(q)==h,p;hashes[str(q)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 for folder,names in [('clock_row_preparation_v2_446',['all_rows_matrix.npz','row_arrays.npz']),('clock_row_preparation_446',['coordinate_map.npz','equality_matrix.npz','equality_arrays.npz'])]:
  for name in names:hashes[str(D/folder/name)]=sha(D/folder/name)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,compile_actual_exit=0,compile_exit_evidence=compile_exit,dependencies_actual_exit_verified=True,dependency_exits=ex,source_input_sha256=hashes,prior_checks=checks,workers=workers,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,new_initial_complete_lift=True,old450_or462_point_used=False,physical_admission=False)
 (D/'free_gap_path_launch_472.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='source_input_sha256'},indent=2),flush=True)
if __name__=='__main__':main()
