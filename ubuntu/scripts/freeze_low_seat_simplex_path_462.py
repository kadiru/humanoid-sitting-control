"""Freeze ONE new joint conserved-load construction after actual preflight exits."""
from pathlib import Path
import json,datetime,shutil
from low_seat_simplex_solver_462 import D,sha,SPEC
R=Path('/home/kadir/humanoid-sitting');read=lambda p:json.loads(Path(p).read_text())
def main():
 assert not (D/'simplex_path_launch_462.json').exists() and not (D/'simplex_path_462').exists()
 ex=read(D/'simplex_solver_preparation_exits_462.json');assert ex['preparation462_actual_exit']==ex['assembly463_actual_exit']==0
 assert read(D/'simplex_solver_preparation_audit_463/independent_audit.json')['passed']
 checks=[]
 for n in ['prior_hashes.json','simplex_evidence_preservation_461.json','simplex_supplemental_461.json']:
  m=read(D/n)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=n,files=len(m),passed=True))
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 source=['low_seat_simplex_solver_462.py','prepare_low_seat_simplex_solver_462.py','build_low_seat_simplex_path_462.py','audit_low_seat_simplex_assembly_463.py','audit_low_seat_simplex_steps_463.py','audit_low_seat_simplex_path_463.py','audit_low_seat_simplex_result_463.py','freeze_low_seat_simplex_path_462.py','low_seat_simplex_retraction_460.py','audit_low_seat_simplex_v2_461.py','low_seat_collocation_solver_448.py','low_seat_clock_transcription_444.py']
 hashes={str(R/'scripts'/n):sha(R/'scripts'/n) for n in source}
 for folder in ['simplex_preparation_460','simplex_audit_v2_461','simplex_solver_preparation_462','simplex_solver_preparation_audit_463']:
  for p,h in read(D/folder/'SHA256.json').items():
   q=D/folder/p;assert sha(q)==h;hashes[str(q)]=h
  hashes[str(D/folder/'SHA256.json')]=sha(D/folder/'SHA256.json')
 for folder,names in [('clock_row_preparation_v2_446',['all_rows_matrix.npz','global_jacobian.npz','row_arrays.npz']),('clock_row_preparation_446',['coordinate_map.npz','equality_matrix.npz','equality_arrays.npz']),('clock_transcription_preparation_444',['mesh.npz'])]:
  for n in names:hashes[str(D/folder/n)]=sha(D/folder/n)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,preparation462_actual_exit=0,assembly463_actual_exit=0,actual_preflight_exits=ex,source_input_sha256=hashes,prior_checks=checks,workers=workers,specification=SPEC,disk_free_bytes=shutil.disk_usage(D).free,new_algorithm=True,source440_infeasible_seed=True,old450_or456_point_used=False,physical_admission=False)
 (D/'simplex_path_launch_462.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='source_input_sha256'},indent=2),flush=True)
if __name__=='__main__':main()
