"""Correct only offline auxiliary contact labels; reuse all original calculations."""
from pathlib import Path
import json,hashlib,shutil,datetime
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'spatial_wrench_diagnosis_255';O=D/'spatial_wrench_diagnosis_v2_255';R=D/'spatial_contact_mapping_recovery_255'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 R.mkdir(exist_ok=False);O.mkdir(exist_ok=False)
 for p in [Path('scripts/diagnose_low_seat_spatial_wrench_255.py'),D/'spatial_wrench_255.log',P/'results.json',Path('scripts/audit_low_seat_spatial_wrench_256.py')]:
  if p.exists():shutil.copy2(p,R/p.name)
 r=json.loads((P/'results.json').read_text());before=json.loads((P/'results.json').read_text());t=np.load(D/'drive_diagnostic_246/every_tick.npz');names=json.loads((D/'drive_diagnostic_246/contact_body_names.json').read_text());ids=[names.index(s+'_'+j+'_link') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch']];changed=[]
 for k,row in enumerate(r['rows']):
  ti=round(row['t']/.02)-1;c=row['case'];value=bool(np.any(t['unexpected_body_normal'][ti,c,ids]>10))
  if value!=row['other_leg_contact']:changed.append(k)
  row['other_leg_contact']=value
 for a,b in zip(before['rows'],r['rows']):
  assert {k:v for k,v in a.items() if k!='other_leg_contact'}=={k:v for k,v in b.items() if k!='other_leg_contact'}
 r['auxiliary_contact_label_correction']=dict(changed_rows=changed,scope='Use saved contact_body_names rather than actuator body order ONLY for offline other_leg_contact labels. Wrench matrices/estimates/force errors/sensitivity/conditions unchanged; no calculation or physics rerun.')
 r['source_sha256'].update({str(p.resolve()):sha(p) for p in [Path(__file__),P/'results.json',D/'drive_diagnostic_246/contact_body_names.json']})
 (O/'results.json').write_text(json.dumps(r,indent=2));shutil.copy2(P/'matrices.npz',O/'matrices.npz');assert sha(P/'matrices.npz')==sha(O/'matrices.npz')
 (R/'recovery.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_preserved=True,change=r['auxiliary_contact_label_correction'],physics_rerun=False,arithmetic_rerun=False,criteria_changed=False),indent=2))
 for d in [O,R]:(d/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir() if p.is_file()},indent=2))
 print(json.dumps(dict(changed_auxiliary_rows=len(changed),matrices_exact=True)))
if __name__=='__main__':main()
