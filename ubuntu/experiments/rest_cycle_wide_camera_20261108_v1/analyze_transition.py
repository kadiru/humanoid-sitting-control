"""Targeted saved-record attribution; no solver, controller or simulator."""
import os,time,resource,json,hashlib
from pathlib import Path
START=time.monotonic()
os.environ['CUDA_VISIBLE_DEVICES']=''
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
import numpy as np
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent
O=R/'outputs'/H.name/'sit_coffee_wide'
CONTACTS=['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']


def main():
    with np.load(O/'trajectory.npz') as z:d=dict(z)
    with np.load(O/'substeps.npz') as z:s=dict(z)
    sg=json.loads((O/'controller_signals.json').read_text())
    meta=json.loads((O/'results.json').read_text())
    active=np.linalg.norm(d['physical_force_w'][:,0],axis=-1)>.01*d['physical_bodyweight_N'][:,0,None]
    slips=active&(d['physical_slip_m_s'][:,0]>.02)
    assert np.array_equal(slips.any(1),~d['no_slip'][:,0])
    rows=[]
    for i,name in enumerate(CONTACTS):
        ids=np.flatnonzero(slips[:,i]);first=float((ids[0]+1)*.02) if len(ids) else None
        last=float((ids[-1]+1)*.02) if len(ids) else None
        rows.append(dict(body=name,slip_control_samples=len(ids),first_s=first,last_s=last,
            maximum_loaded_slip_m_s=float(np.max(np.where(active[:,i],d['physical_slip_m_s'][:,0,i],0)))))
    ids=np.flatnonzero(slips.any(1))
    samples={}
    for tick in sorted(set([0,349,649,999]+[int(i) for i in ids[[0,-1]]])):
        v=sg[tick];speed=abs(d['qd_pre'][tick,0,:23]);order=np.argsort(speed)[-3:][::-1]
        samples[str(tick+1)]=dict(time_s=(tick+1)*.02,phase=v['phase'],quiet=v['quiet'],
            pitch_error_deg=float(np.rad2deg(v['pitch']-v['pitch_reference'])),
            geometry_valid=v['measured_geometry_valid'],geometry_age_s=v['geometry_age_s'],
            normal_error=v['normal_error'],offset_error_m=v['offset_error_m'],
            highest_joint_speeds=[dict(joint=meta['joint_names'][j],rad_s=float(speed[j])) for j in order],
            support_bw=float(d['support_bw'][tick,0]))
    forces=s['normal']+s['friction']
    avg=forces.reshape(1000,4,7,3).mean(1)
    last=forces.reshape(1000,4,7,3)[:,-1]
    recorded=d['physical_force_w'][:,0]
    bw=float(d['physical_bodyweight_N'][-1,0])
    foot_changes={}
    for name in CONTACTS[-2:]:
        idx=meta['body_names'].index(name)
        foot_changes[name]=(d['body'][-1,0,idx,:3]-d['body'][0,0,idx,:3]).tolist()
    result=dict(physical_success=False,final_hold_s=6.46,slip_samples=int(len(ids)),
        contact_slip_attribution=rows,diagnostic_samples=samples,
        camera_valid_control_samples=int(sum(v['measured_geometry_valid'] for v in sg)),
        maximum_camera_age_s=max(v['geometry_age_s'] for v in sg),
        phase_counts={p:sum(v['phase']==p for v in sg) for p in sorted({v['phase'] for v in sg})},
        ankle_link_world_displacement_m=foot_changes,
        reported_force_last_substep_max_difference_N=float(abs(last-recorded).max()),
        final5s_mean_reported_control_net_force_N=(recorded[-250:].sum(1).mean(0)-[0,0,bw]).tolist(),
        final5s_mean_four_substep_net_force_N=(avg[-250:].sum(1).mean(0)-[0,0,bw]).tolist(),
        force_scope='Clock alignment audit only. Does not change success flags, certify motor torque, validate a load estimator, or prove the cause of an older run without its substeps.',
        physical_steps_added=0,model_solves=0,PPO_updates=0,wall_seconds=time.monotonic()-START,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [O/'trajectory.npz',O/'substeps.npz',O/'controller_signals.json',Path(__file__)]})
    with (O/'transition_analysis.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','diagnostic_samples']}))
    print(json.dumps(samples))

if __name__=='__main__':main()
