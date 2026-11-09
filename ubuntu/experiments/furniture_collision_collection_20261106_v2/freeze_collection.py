import sys,json,ast,hashlib,time
from pathlib import Path
root=Path('/home/kadir/humanoid-sitting');exp=root/'experiments';here=Path(__file__).resolve().parent;old=exp/'furniture_collision_collection_20261106_v1';v3=exp/'furniture_load_repair_20261106_v3';modeldir=exp/'furniture_repair_20261106_v1'
sys.path[:0]=[str(here),str(v3),str(modeldir/'model_v4'),str(modeldir),str(root/'scripts'),str(root/'sitting_task/src')]
import numpy as np
from json_types import encode
from seat_plan import SeatPlan
from humanoid_sitting.shared_frame import BodyModel
from audit_seat_plan import audit
from audit_complete_path import audit as audit_commands
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
model=BodyModel(root/'runs/g1_kinematics_geometry_04/model.json');inertia=read(Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json'));shapes=read(exp/'furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
jobs=read(old/'calibration_jobs.json');validation=[]
for j in jobs[:2]:
 p=Path(j['output']);assert read(p/'failure.json')['controls']==0
 z=dict(np.load(p/'frame_0.npz'));assert read(p/'startup.json')['chosen_slots']==[0]
 # FullBodyEnv.reset uses bank command unchanged for all25 warmup controls.
 with np.load(exp/'furniture_examples_20261105_v1/banks'/j['source_case']/'reset_candidates.npz') as init:previous=init['command'][0,0]
 plan=SeatPlan(model,inertia,shapes,z['q'],z['gravity'],z['geometry'],z['selected_points'],previous)
 construction=read(Path(j['model_directory'])/(j['source_case']+'.plan.json'));path=read(Path(j['model_directory'])/(j['source_case']+'.complete.json'))
 result=dict(geometry=audit(plan,construction),commands=audit_commands(plan,construction,path))
 plain=json.loads(json.dumps(result,default=encode,allow_nan=False));passed=all(v for k,v in plain['geometry']['tests'].items() if k!='time_for_original_hold') and plain['commands']['passed']
 validation.append(dict(case=j['name'],saved_startup_only=True,active_controls=0,serialized_exact_native_scalars=True,admitted=passed,checks=plain))
(here/'saved_reset_serialization_audit.json').write_text(json.dumps(validation,indent=2))
print(json.dumps([dict(case=v['case'],admitted=v['admitted'],failed={group:[k for k,b in v['checks'][group]['tests'].items() if not b] for group in ['geometry','commands']}) for v in validation]),flush=True)
rejected={v['case'] for v in validation if not v['admitted']}
jobs=[j for j in jobs if j['name'] not in rejected]
for i,j in enumerate(jobs):
 j['script']=str(here/('legacy_demo.py' if j['script'].endswith('furniture_load_repair_20261106_v3/calibration_demo.py') else 'calibration_demo.py'));j['output']=str(root/'outputs'/here.name/'calibration_demos'/j['name'])
 assert not Path(j['output']).exists()
(here/'calibration_jobs.json').write_text(json.dumps(jobs,indent=2))
prior=read(old/'calibration_freeze.json');files=prior['files'].copy()
for n,h in files.items():assert sha(n)==h,('PRIOR_FROZEN_SOURCE_CHANGED',n)
for p in here.iterdir():
 if p.suffix=='.py':ast.parse(p.read_text())
 if p.is_file():files[str(p)]=sha(p)
review=dict(passed=True,json_scalar_native_value_validation=True,two_saved_reset_results_retained=True,rejected_camera_hull_starts_excluded=True,old_action_controller_byte_identical=sha(here/'legacy_normal_landing.py')==sha(v3/'normal_landing.py'),unchanged_acceptance_auditor=sha(here/'audit_rollout.py')==sha(v3/'audit_rollout.py'),same_models_and_inputs=True,abort_on_software_error=True,physical_success_claim=False)
assert review['old_action_controller_byte_identical'] and review['unchanged_acceptance_auditor']
(here/'source_review.json').write_text(json.dumps(review,indent=2));files[str(here/'source_review.json')]=sha(here/'source_review.json')
with (here/'calibration_freeze.json').open('x') as f:json.dump(dict(files=files,source_review=review,deadline_epoch=prior['deadline_epoch']),f,indent=2)
print('frozen',len(files),'remaining_hours',(prior['deadline_epoch']-time.time())/3600)
