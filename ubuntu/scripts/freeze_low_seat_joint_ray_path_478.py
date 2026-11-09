"""Freeze ONE distinct shared-affine-ray run, inherited INITIAL and independent auditors."""
from pathlib import Path
import json,datetime,shutil
from low_seat_joint_ray_execution_478 import D,sha,SPEC
R=Path('/home/kadir/humanoid-sitting');read=lambda p:json.loads(Path(p).read_text())
def main():
 assert not (D/'joint_ray_path_launch_478.json').exists() and not (D/'joint_ray_path_478').exists()
 ex={name:read(D/name) for name in ['joint_ray_preparation_exit_476.json','joint_ray_audit_exit_477.json']}
 assert all(x['actual_exit']==0 for x in ex.values())
 compile_exit=read(D/'joint_ray_compile_exit_478.json');assert compile_exit['actual_exit']==0
 assert read(D/'joint_ray_audit_477/independent_audit.json')['passed']
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_475.json','supplemental_preservation_475.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 sources=['low_seat_joint_ray_476.py','prepare_low_seat_joint_ray_476.py','audit_low_seat_joint_ray_477.py','low_seat_joint_ray_execution_478.py','build_low_seat_joint_ray_path_478.py','audit_low_seat_joint_ray_steps_479.py','audit_low_seat_joint_ray_path_479.py','audit_low_seat_joint_ray_result_479.py','freeze_low_seat_joint_ray_path_478.py','low_seat_free_gap_execution_472.py','low_seat_free_gap_solver_470.py','low_seat_exact_gap_chart_468.py','low_seat_gap_coordinates_v2_467.py','low_seat_simplex_retraction_460.py','low_seat_collocation_solver_448.py','low_seat_clock_transcription_444.py']
 hashes={str(R/'scripts'/name):sha(R/'scripts'/name) for name in sources}
 for folder in ['joint_ray_preparation_476','joint_ray_audit_477','free_gap_path_472','free_gap_path_audit_473','exact_gap_chart_preparation_468','exact_gap_chart_audit_v2_469','free_gap_solver_preparation_470','free_gap_assembly_audit_471','gap_coordinate_preparation_v3_466','contact_lift_440','contact_lift_audit_441','clock_transcription_preparation_444']:
  for p,h in read(D/folder/'SHA256.json').items():
   q=D/folder/p;assert sha(q)==h,p;hashes[str(q)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 for folder,names in [('clock_row_preparation_v2_446',['all_rows_matrix.npz','row_arrays.npz']),('clock_row_preparation_446',['coordinate_map.npz','equality_matrix.npz','equality_arrays.npz'])]:
  for name in names:hashes[str(D/folder/name)]=sha(D/folder/name)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,compile_actual_exit=0,compile_exit_evidence=compile_exit,dependencies_actual_exit_verified=True,dependency_exits=ex,source_input_sha256=hashes,prior_checks=checks,workers=workers,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,new_initial_complete_lift=False,seed_is_sealed472_initial=True,initial_root_repeated=False,old450_or462_point_used=False,physical_admission=False)
 (D/'joint_ray_path_launch_478.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='source_input_sha256'},indent=2),flush=True)
if __name__=='__main__':main()
