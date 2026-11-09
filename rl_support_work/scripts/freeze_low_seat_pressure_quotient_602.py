"""Freeze one exact pressure-quotient necessary test and independent audit."""
from low_seat_pressure_quotient_602 import *
import datetime
def main():
 assert not (D/'pressure_quotient_launch_602.json').exists()
 ce=read(D/'pressure_quotient_compile_exit_602.json');assert ce['actual_exit']==0
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_601.json','supplemental_preservation_601.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 ws=workers(Path(__file__).name);assert not ws,ws
 names=['low_seat_pressure_quotient_602.py','freeze_low_seat_pressure_quotient_602.py','prepare_low_seat_pressure_quotient_602.py','audit_low_seat_pressure_quotient_603.py']
 paths=[R/'scripts'/n for n in names]+[D/n for n in ['zero_face_preparation_572/equality.npz','zero_face_preparation_572/equality_original_positive_ids.npy','zero_face_preparation_572/arrays.npz','gap_coordinate_preparation_v3_466/coordinate_basis.npz','gap_coordinate_preparation_v3_466/layout.json','clock_transcription_preparation_444/mesh.npz','closure_record_601.json','supplemental_preservation_601.json','pressure_quotient_compile_exit_602.json','pressure_quotient_authoring_602.json']]
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),specification=SPEC,prior_checks=checks,workers=ws,compile_actual_exit=0,source_input_sha256={str(p):sha(p) for p in paths},scientific_execution_started=False)
 (D/'pressure_quotient_launch_602.json').write_text(json.dumps(r,indent=2))
 print(json.dumps({k:r[k] for k in ['utc','prior_checks','workers','compile_actual_exit']},indent=2))
if __name__=='__main__':main()
