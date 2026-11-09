"""Freeze ONE NEW exact-implication full-system witness run and independent auditors; no492 restart."""
from pathlib import Path
import json,datetime,shutil
from low_seat_quotient_execution_518 import D,sha,SPEC
R=Path('/home/kadir/humanoid-sitting');read=lambda p:json.loads(Path(p).read_text())
def main():
 assert not (D/'quotient_path_launch_518.json').exists() and not (D/'quotient_path_518').exists()
 ex={n:read(D/n) for n in ['exact_row_quotient_finalization_exit_516.json','exact_row_quotient_audit_exit_517.json']};assert all(x['actual_exit']==0 for x in ex.values())
 assert read(D/'exact_row_quotient_preparation_exit_516.json')['actual_exit']==1
 assert read(D/'exact_row_quotient_audit_517/independent_audit.json')['passed']
 compile_exit=read(D/'quotient_compile_exit_518.json');assert compile_exit['actual_exit']==0
 assert read(D/'lifted_feasibility_audit_491/independent_audit.json')['passed']
 assert read(D/'mixed_coordinate_audit_505/independent_audit.json')['passed']
 checks=[]
 for n in ['prior_hashes.json','exact_row_quotient_evidence_preservation_517.json','exact_row_quotient_supplemental_517.json']:
  m=read(D/n)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=n,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 sources=['low_seat_mixed_execution_506.py','low_seat_exact_row_quotient_v2_516.py','low_seat_quotient_execution_518.py', 'solve_low_seat_quotient_lp_518.py', 'build_low_seat_quotient_path_518.py', 'freeze_low_seat_quotient_path_518.py', 'audit_low_seat_quotient_steps_519.py', 'audit_low_seat_quotient_path_519.py', 'audit_low_seat_quotient_result_519.py', 'low_seat_lifted_execution_492.py', 'low_seat_lifted_feasibility_490.py', 'low_seat_mixed_coordinates_504.py', 'low_seat_feature_collocation_488.py', 'audit_low_seat_lifted_feasibility_491.py', 'low_seat_exact_gap_chart_468.py', 'low_seat_clock_transcription_444.py']
 hashes={str(R/'scripts'/n):sha(R/'scripts'/n) for n in sources}
 for n in ['exact_row_quotient_preparation_516','exact_row_quotient_audit_517','mixed_coordinate_preparation_504','mixed_coordinate_audit_505','sparse_gap_residual_preparation_502','sparse_gap_residual_audit_503','residual_coordinate_preparation_496','residual_coordinate_audit_497','lifted_feasibility_preparation_490','feature_collocation_preparation_488','lifted_feasibility_audit_491','free_gap_path_472','free_gap_path_audit_473','joint_ray_path_audit_479','contact_lift_440','contact_lift_audit_441','clock_transcription_preparation_444','gap_coordinate_preparation_v3_466','exact_gap_chart_preparation_468']:
  for p,h in read(D/n/'SHA256.json').items():
   q=D/n/p;assert sha(q)==h,p;hashes[str(q)]=h
  hashes[str(D/n/'SHA256.json')]=sha(D/n/'SHA256.json')
 for n,names in [('clock_row_preparation_v2_446',['all_rows_matrix.npz','row_arrays.npz']),('clock_row_preparation_446',['coordinate_map.npz','equality_matrix.npz','equality_arrays.npz'])]:
  for name in names:hashes[str(D/n/name)]=sha(D/n/name)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,compile_actual_exit=0,compile_exit_evidence=compile_exit,dependencies_actual_exit_verified=True,dependency_exits=ex,source_input_sha256=hashes,prior_checks=checks,workers=workers,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,new_initial_complete_lift=False,seed_is_sealed472_initial=True,initial_root_repeated=False,old478_final_or_rejected_witness_used=False,primary_directions_never_decoded=True,physical_admission=False)
 (D/'quotient_path_launch_518.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='source_input_sha256'},indent=2),flush=True)
if __name__=='__main__':main()
