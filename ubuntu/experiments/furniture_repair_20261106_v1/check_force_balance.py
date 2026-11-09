"""Saved-pose momentum check; does not change contact scores or run physics."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/home/kadir/humanoid-sitting')
OUT=ROOT/'outputs/furniture_repair_20261106_v1/motor_axis_readout'
read=lambda p:json.loads(Path(p).read_text())
d=np.load(OUT/'trajectory.npz')
m=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
model=read(ROOT/'runs/g1_kinematics_geometry_04/model.json')
ids=[model['body_names'].index(name) for name in m['body_names']]
pose=d['body'][:,0,ids].astype(float)
local=np.broadcast_to(np.array(m['local_com']),pose[:,:,:3].shape)
coms=pose[:,:,:3]+Rotation.from_quat(pose[:,:,3:7].reshape(-1,4)).apply(local.reshape(-1,3)).reshape(local.shape)
mass=np.array(m['mass']);total=mass.sum();com=np.einsum('tbc,b->tc',coms,mass)/total
dt=.02;force=d['physical_force_w'][:,0].sum(1).astype(float)
net=force-np.array([0.,0.,total*9.81])
raw=read(OUT/'raw_contact_buffers.json')[-1]
channels=[]
for i,c in enumerate(raw['channels']):
    a=c['tuple_components_0_to_3']
    normal=(np.array(a[0]).reshape(-1,1)*np.array(a[2]).reshape(-1,3)).sum(0)
    aggregate=np.array(raw['aggregate_force_w_N'])[0,i]
    channels.append(dict(body=c['body'],normal_N=normal.tolist(),resultant_N=aggregate.tolist(),
                         inferred_friction_N=(aggregate-normal).tolist(),contact_count=c['count']))
# Average endpoint velocity from 10 controls each, to suppress float32 differentiation.
v_start=(com[760]-com[750])/(10*dt);v_end=(com[999]-com[989])/(10*dt)
duration=(994-755)*dt
measured_impulse=net[755:994].sum(0)*dt
pose_impulse=total*(v_end-v_start)
result=dict(backend='physx (confirmed by saved launch log; not ovphysx)',mass_kg=float(total),
    final5s_mean_total_force_N=force[-250:].mean(0).tolist(),
    final5s_mean_net_force_N=net[-250:].mean(0).tolist(),
    final5s_COM_range_m=np.ptp(com[-250:],axis=0).tolist(),
    endpoint_window_duration_s=duration,reported_contact_impulse_Ns=measured_impulse.tolist(),
    pose_momentum_change_Ns=pose_impulse.tolist(),impulse_discrepancy_Ns=(measured_impulse-pose_impulse).tolist(),
    final5s_max_wrong_force_N=float(d['wrong_force'][-250:].max()),
    final_channels=channels,
    conclusion='Saved contact reporting and pose momentum disagree. These data do not identify the faulty producer. Do not fit a force correction to this discrepancy.',
    limitations=['Pose finite differences are at control rate, not physics substeps.',
                 'Raw friction anchors and all-substep incoming wrenches were not saved.',
                 'No scoring correction or physical-success reclassification is justified.'],
    simulator_steps=0,ppo_updates=0,
    trajectory_sha256=hashlib.sha256((OUT/'trajectory.npz').read_bytes()).hexdigest())
with (OUT/'force_balance_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
