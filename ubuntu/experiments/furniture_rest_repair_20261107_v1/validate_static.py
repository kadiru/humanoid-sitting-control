import json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sitting_task/src'))
from humanoid_sitting.shared_frame import BodyModel
from static_support import StaticSupport
m=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json');inertia=json.loads(Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json').read_text())
shapes=json.loads((ROOT/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json').read_text())['shapes']
mesh={b:np.concatenate([s['vertices'] for s in shapes if s['body']==b]) for b in {s['body'] for s in shapes}}
model=StaticSupport(m,inertia,mesh);rows=[]
for p in sorted((ROOT/'outputs/furniture_camera_coverage_20261106_v1/calibration_demos').iterdir()):
 plan=json.loads((p/'plan.json').read_text());nodes=plan['nodes'];checks=[]
 for i in [0,len(nodes)//2,len(nodes)-1]:
  n=nodes[i];q=np.array(n['q']);root=np.array(n['root']);bias,r=model.evaluate(q,root);check=model.audit_virtual_work(q,root,r)
  checks.append(dict(node=i,report=r,audit=check,ankle_roll_bias_rad=[float(bias[m.names.index(s+'_ankle_roll_joint')]) for s in ['left','right']]))
 rows.append(dict(case=p.name,checks=checks))
(HERE/'static_feedforward_validation.json').write_text(json.dumps(dict(rows=rows,physics_steps=0,mechanism='Previously omitted lateral joint gravity/contact moments now included; no gain or action-limit change',physical_benefit_unproven=True),indent=2));print(json.dumps([dict(case=r['case'],ankle_roll_bias=[x['ankle_roll_bias_rad'] for x in r['checks']],max_audit_error=max(x['audit']['maximum_analytic_vs_virtual_work_Nm'] for x in r['checks'])) for r in rows]))
