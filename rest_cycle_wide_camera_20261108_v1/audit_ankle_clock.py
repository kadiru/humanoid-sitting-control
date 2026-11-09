"""Saved native link angular velocities versus recorded joint velocity.

No finite-difference velocity is substituted into control or success scoring.
Within-control projections use the end-of-control parent orientation and are
labelled approximate; the final substep has a matching measured parent pose.
"""
import os,sys,time,resource,json,hashlib
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']=''
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'sitting_task/src'))
from humanoid_sitting.shared_frame import BodyModel
O=R/'outputs'/H.name/'sit_coffee_wide'


def main():
    with np.load(O/'trajectory.npz') as z:d=dict(z)
    with np.load(O/'substeps.npz') as z:s=dict(z)
    meta=json.loads((O/'results.json').read_text());model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json')
    joint_map={j['name']:j for j in model.joints};rows=[]
    for side in ['left','right']:
        name=side+'_ankle_roll_joint';joint=joint_map[name];index=meta['joint_names'].index(name)
        parent=meta['body_names'].index(joint['body0'][0]);child=meta['body_names'].index(joint['body1'][0])
        parent_R=Rotation.from_quat(d['body'][:,0,parent,3:7]).as_matrix()
        local_axis=model.frames[name][0][:3,'XYZ'.index(joint['axis'])]
        axis=np.einsum('tij,j->ti',parent_R,local_axis)
        angular=s['velocity'][:,0,child,3:6]-s['velocity'][:,0,parent,3:6]
        projected=np.sum(angular.reshape(1000,4,3)*axis[:,None,:],axis=-1)
        raw=d['physical_joint_velocity'][:,0,index]
        steady=projected[-250:];q=d['q'][:,0,index]
        rows.append(dict(joint=name,
            final_substep_projection_vs_recorded_joint_velocity_max_error_rad_s=float(np.max(abs(projected[:,-1]-raw))),
            final5s_raw_control_velocity_mean_rad_s=float(raw[-250:].mean()),
            final5s_projected_substep_phase_means_rad_s=steady.mean(0).tolist(),
            final5s_approximate_all_substep_velocity_mean_rad_s=float(steady.mean()),
            final5s_approximate_all_substep_velocity_rms_rad_s=float(np.sqrt(np.mean(steady**2))),
            final5s_joint_position_change_rad=float(q[-1]-q[-251]),
            final5s_joint_position_range_rad=[float(q[-250:].min()),float(q[-250:].max())],
            final5s_approximate_velocity_integral_rad=float(steady.sum()*.005)))
    result=dict(rows=rows,scope='Aligned native readout diagnosis; intermediate parent rotations approximate, no corrected velocity/gate/score/physical claim.',
        physics_steps=0,model_solves=0,wall_seconds=time.monotonic()-START,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [O/'trajectory.npz',O/'substeps.npz',R/'runs/g1_kinematics_geometry_04/model.json',Path(__file__)]})
    with (O/'ankle_clock_audit.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))

if __name__=='__main__':main()
