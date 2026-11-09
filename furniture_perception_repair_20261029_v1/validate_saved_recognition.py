"""Saved-frame development test; exact current sensor records are collected in the fresh test."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/home/kadir/humanoid-sitting');sys.path.insert(0,str(ROOT/'sitting_task/src'))
from humanoid_sitting.shared_frame import BodyModel
from perception import observe

def main():
    model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json');T=np.eye(4);T[:3,:3]=[[0,2**-.5,-2**-.5],[1,0,0],[0,-2**-.5,-2**-.5]];T[:3,3]=[-.07,0,.38]
    cases=json.loads((ROOT/'experiments/furniture_test_20261028_v1/cases.json').read_text());rows=[]
    for case in cases:
        z=np.load(ROOT/'outputs/motion_prior_ppo_20261029_v1'/('depth_'+case['id'])/'frame_0.npz');b=np.load(ROOT/'outputs/full_body_contact_imitation_v4/dataset'/case['task']/'bank.npz')
        q=b['reset_q'][0];root=b['reset_root_pose'][0];R=Rotation.from_quat(root[3:]).as_matrix();gravity=R.T@np.array([0.,0.,-1.])
        intent='back' if case['approach']=='rear_back' else 'arm' if case['approach']=='arm' else 'seat' if case['approach'].startswith('seat') else 'top'
        value,diag=observe(model,z['K'],T,z['depth'],q,gravity,intent)
        # Ground truth enters only this comparison, not observe().
        normal=R.T@np.asarray(case['target_normal']);offset=(root[:3]-case['target_center'])@np.asarray(case['target_normal'])
        angle=float(np.rad2deg(np.arccos(np.clip(value[2:5]@normal,-1,1)))) if value is not None else None
        error=float(value[5]-offset) if value is not None else None
        passed=bool(value is not None and angle<=5 and abs(error)<=.015)
        row=dict(case=case['id'],passed=passed,normal_error_deg=angle,offset_error_m=error,diagnostics=diag);rows.append(row);print(case['id'],passed,angle,error)
    out=ROOT/'outputs/furniture_perception_repair_20261029_v1';out.mkdir(exist_ok=True)
    (out/'saved_recognition.json').write_text(json.dumps(dict(rows=rows,passed=sum(x['passed'] for x in rows),total=14,scope='Development arithmetic using saved depth and reset encoders/IMU; fresh live sensor verification still required.'),indent=2)+'\n')

if __name__=='__main__':main()
