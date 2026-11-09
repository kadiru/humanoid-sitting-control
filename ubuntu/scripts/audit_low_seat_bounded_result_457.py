"""Conditional result audit: new final states get new checks; exact old input
inherits sealed441 geometry/work without repeating that completed model loop.
The frozen full-state auditor remains unchanged and is invoked only if needed.
"""
from pathlib import Path
import json,datetime
import numpy as np
from scipy import sparse
from low_seat_clock_transcription_444 import D,sha
from audit_low_seat_bounded_steps_457 import audit_steps
O=D/'bounded_path_audit_457';read=lambda p:json.loads(Path(p).read_text())
def main():
 assert read(D/'bounded_path_exit_456.json')['actual_exit']==0
 for p,h in read(D/'bounded_path_456/SHA256.json').items():assert sha(D/'bounded_path_456'/p)==h,p
 f=np.load(D/'bounded_path_456/final.npz');i=np.load(D/'bounded_path_456/initial.npz')
 if not (np.array_equal(f['X'],i['X']) and np.array_equal(f['Y'],i['Y'])):
  from audit_low_seat_bounded_path_457 import main as full
  full();return
 O.mkdir(exist_ok=False);step=audit_steps(O);mesh=np.load(D/'clock_transcription_preparation_444/mesh.npz');pr=np.load(D/'clock_transcription_preparation_444/primal_replay.npz');row=np.load(D/'clock_row_preparation_v2_446/row_arrays.npz')
 assert np.array_equal(f['X'],row['features'][:,:24]);assert np.array_equal(f['Y'],row['features'][:,24:]);assert np.array_equal(f['X'][mesh['label_to_state']],pr['X'])
 source=read(D/'contact_lift_440/results.json');audit=read(D/'contact_lift_audit_441/independent_audit.json');assert audit['passed'];assert not audit['sampled_geometry_failures']
 for folder in ['contact_lift_440','contact_lift_audit_441','clock_transcription_preparation_444']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p
 records=read(D/'contact_lift_440/records.json');geometry=read(D/'contact_lift_audit_441/geometry.json');samples=[]
 for j,(g,r) in enumerate(zip(geometry,records)):
  samples.append(dict(record=j,time_s=r['time_s'],phase=r['phase'],orientation_phase=r['orientation_phase'],housing_m=g['clearance_m'],cop_margin_m=g['foot_cop_margin_m'],gap_m=g['pelvis_gap_m'],floor_m=g['floor_clearance_m'],patch_m=g['patch_margin_m'],foot_position_m=g['foot_position_error_m'],foot_orientation_rad=g['foot_orientation_error_rad'],sampled_geometry_passed=r['sampled_geometry_passed'],evidence='inherited sealed441 after exact final/source identity'))
 (O/'geometry.json').write_text(json.dumps(samples));old=sparse.load_npz(D/'clock_row_preparation_446/original_rows_matrix.npz');raw=old@np.c_[f['X'],f['Y']].ravel()+row['old_constant'];prior=raw-row['old_tightening'];s=np.load(D/'contact_lift_440/states.npz');clockraw=s['clock_raw'];clockprior=s['clock_prior']
 np.savez_compressed(O/'independent_rows.npz',old_raw=raw,old_prior=prior,clock_raw=clockraw,clock_prior=clockprior)
 identity=dict(final_X_exact_initial=True,final_Y_exact_initial=True,final_features_exact_saved444=True,source440_manifest_sha256=sha(D/'contact_lift_440/SHA256.json'),source441_manifest_sha256=sha(D/'contact_lift_audit_441/SHA256.json'),preparation444_manifest_sha256=sha(D/'clock_transcription_preparation_444/SHA256.json'),new_full_geometry_records=0,new_finite_work_columns=0,details='All1467 geometry/work records inherit sealed441; no completed geometry/root/finite-work loop repeated. Independent trial and row audits are new saved-array calculations.')
 (O/'inheritance.json').write_text(json.dumps(identity,indent=2))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,passed=True,arithmetic_only=True,records=1467,new_full_geometry_records=0,new_finite_work_columns=0,inherited441_exact_records=1467,duplicate_new_records=0,maximum_errors=audit['maximum_errors'],maximum_errors_scope='inherited441, not new finite-work measurements',sampled_geometry_passed=True,sampled_geometry_failures=[],minimum_housing_m=source['minimum_housing_m'],minimum_cop_m=source['minimum_support_m'],moving_command_margin_rad=source['minimum_command_margin_rad'],maximum_static_effort_utilization=source['maximum_effort_utilization'],original_clock_rate_minimum=float(clockraw[:,:32].min()),original_clock_target_minimum=float(clockraw[:,32:].min()),prior_clock_rate_minimum=float(clockprior[:,:32].min()),prior_clock_target_minimum=float(clockprior[:,32:].min()),step_audit=step,inheritance=identity,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
