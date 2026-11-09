"""Conditional NEW532 result audit; unchanged initial states inherit sealed473.
No completed geometry/finite-work loop is repeated on an identical export.
"""
from pathlib import Path
import json,hashlib,datetime,shutil
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'schur_path_audit_533'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert read(D/'schur_path_exit_532.json')['actual_exit']==0
 for p,h in read(D/'schur_path_launch_532.json')['source_input_sha256'].items():assert sha(p)==h,p
 for folder in ['schur_path_532','free_gap_path_472','free_gap_path_audit_473','contact_lift_440','contact_lift_audit_441','clock_transcription_preparation_444']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p
 a=np.load(D/'schur_path_532/final.npz');b=np.load(D/'free_gap_path_472/initial.npz')
 if not (np.array_equal(a['X'],b['X']) and np.array_equal(a['Y'],b['Y'])):
  from audit_low_seat_schur_path_533 import main as full
  full();return
 O.mkdir(exist_ok=False)
 from audit_low_seat_schur_steps_533 import audit_steps
 step=audit_steps(O);prior=read(D/'free_gap_path_audit_473/independent_audit.json')
 for name in ['independent_states.npz','independent_rows.npz','geometry.json']:shutil.copy2(D/'free_gap_path_audit_473'/name,O/name)
 out=dict(prior);out.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,step_audit=step,new_finite_work_columns=0,inherited473_exact_records=1467,inherited441_exact_records=0,duplicate_new_records=0,all_geometry_and_work_inherited=True,inherited_error_scope='Sealed473 aggregate over205 new473 states plus1262 inherited441 states; no new533 geometry/work computation',source_audit_sha256=sha(D/'free_gap_path_audit_473/independent_audit.json'),source_manifest_sha256=sha(D/'free_gap_path_audit_473/SHA256.json'),model_calls=0,root_calls=0,geometry_calls=0,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
