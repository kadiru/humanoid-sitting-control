"""Explain catalog safety stops from saved poses, loads and guard flags."""
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from furniture_skill_comparison import STUDY as CATALOG, SIZES
from off_axis_sitting_study import STUDY
from run import read, sha, write


MODEL = Path('/home/kadir/humanoid-sitting/runs/g1_kinematics_geometry_com_01/model.json')
FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')
GUARDS = ('fall', 'forbidden_contact', 'torque', 'impact', 'joint_limit')
DT = .02


def plot_example(t, over, seat, feet, stop):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (balance, load) = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, layout='constrained')
    fig.suptitle('Armchair at 88% scale', x=.055, ha='left', fontsize=17, weight='bold')
    balance.axhline(0, color='#899ca6', linewidth=1)
    balance.plot(t, over*100, color='#1a5d84', linewidth=2)
    balance.fill_between(t, 0, over*100, where=over>0, color='#cf684c', alpha=.21)
    balance.set_ylabel('Center of mass relative\nto the front of the feet (cm)')
    balance.annotate(f'{over[-1]*100:.1f} cm ahead of the toes', (stop,over[-1]*100),
                     xytext=(-10,12), textcoords='offset points', ha='right', color='#963d2f')
    load.plot(t,feet.sum(1)*100,color='#465e6c',label='Both feet',linewidth=1.8)
    load.plot(t,seat*100,color='#25977a',label='Usable seat contact',linewidth=2.3)
    load.set_ylim(-5,None)
    load.set_ylabel('Vertical load (% of body weight)')
    load.set_xlabel('Time from start (s)')
    load.legend(loc='best',frameon=False)
    for axis in (balance,load):
        axis.axvline(stop,color='#b65341',linestyle='--',linewidth=1.2)
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y', alpha=.17)
    load.text(stop,-.25,'Ankle safety stop',transform=load.get_xaxis_transform(),ha='right',color='#963d2f')
    fig.savefig(STUDY/'forward_fall_armchair_88.png',dpi=160)
    plt.close(fig)


def diagnose(folder, names, scale, calibration):
    ev = folder/'evaluation'
    result = read(ev/'results.json')
    meta = read(folder/'bank'/'metadata.json')
    trace = ev/'trajectory.npz'
    audit = read(ev/'independent_audit.json')
    assert audit['passed'] and sha(trace)==audit['files']['trajectory.npz']
    body_names = result['body_names']
    indices = [calibration['body_names'].index(k) for k in body_names]
    mass = np.asarray(calibration['body_mass'])[indices]
    local_com = np.asarray(calibration['body_com_local'])[indices]
    foot_ids = [body_names.index(k) for k in FEET]
    sole = [np.asarray(next(c['points'] for c in calibration['collisions'] if c['body']==k)) for k in FEET]
    assert all(v.shape==(8,3) for v in sole)
    joints = [result['joint_names'].index(meta['joint_names'][i]) for i in meta['joint_ids']]
    lower, upper = np.asarray(meta['lower']),np.asarray(meta['upper'])
    reports = []
    with np.load(trace,allow_pickle=False) as z:
        flags, body, q, forces = (z[k] for k in ('flags','body','q','physical_force_w'))
        valid = z['physical_inside_patch'] & z['physical_rest_patch'] & z['physical_permitted']
        weight = z['physical_bodyweight_N']
        np.testing.assert_allclose(weight[0],mass.sum()*9.81,rtol=.001)
        tilt = z['physical_trunk_tilt_rad']
        foot_speed = z['physical_slip_m_s'][:,:,5:]
        for i,name in enumerate(names):
            bad = np.flatnonzero(flags[:,i].any(1))
            if len(bad) == 0:
                continue
            stop = int(bad[0])
            ticks = slice(0,stop+1)
            pose = body[ticks,i]
            rotation = Rotation.from_quat(pose[...,3:].reshape(-1,4)).as_matrix().reshape(-1,len(body_names),3,3)
            com = ((pose[...,:3]+np.einsum('flij,lj->fli',rotation,local_com))*mass[None,:,None]).sum(1)/mass.sum()
            forward = rotation[0,0,:2,0].copy()
            forward /= np.linalg.norm(forward)
            foot_forces = forces[ticks,i,5:,2]/weight[ticks,i,None]
            foot_active = np.linalg.norm(forces[ticks,i,5:],axis=-1)>.01*weight[ticks,i,None]
            slips = (foot_speed[ticks,i]>.02) & foot_active
            far_toe = []
            for side,index in enumerate(foot_ids):
                corners = pose[:,index,None,:3]+np.einsum('fij,lj->fli',rotation[:,index],sole[side])
                # The front edge of the complete sole is an upper bound on
                # the forward support actually provided by this loaded foot.
                far_toe.append((corners[...,:2]@forward).max(1))
            toe = np.where(foot_forces>.05,np.stack(far_toe,axis=1),-np.inf).max(1)
            over = np.where(np.isfinite(toe),com[:,:2]@forward-toe,np.nan)
            com_speed = np.gradient(com[:,:2]@forward,DT) if stop else np.zeros(1)
            seat = (np.maximum(forces[ticks,i,:5,2],0)*valid[ticks,i,:5]).sum(1)/weight[ticks,i]
            pitch = np.rad2deg(np.arcsin(np.clip(-rotation[:,0,2,0],-1,1)))
            q_stop = q[stop,i,joints]
            joint_margin = np.minimum(q_stop-lower,upper-q_stop)
            tight = int(joint_margin.argmin())
            skew = dict(asset=name,scale_pct=scale,stop_s=round((stop+1)*DT,2),
                        first_guards=[k for k,flag in zip(GUARDS,flags[stop,i],strict=True) if flag],
                        signed_pelvis_pitch_deg=round(float(pitch[-1]),2),
                        pelvis_tilt_deg=round(float(np.rad2deg(tilt[stop,i])),2),
                        legal_seat_peak_pct=round(float(seat.max()*100),2),
                        legal_seat_pct_at_stop=round(float(seat[-1]*100),2),
                        com_ahead_of_loaded_toe_cm=round(float(over[-1]*100),2) if np.isfinite(over[-1]) else None,
                        com_forward_m_s=round(float(np.median(com_speed[-5:])),3),
                        foot_loads_pct=np.round(100*foot_forces[-1],2).tolist(),
                        loaded_foot_slip_ticks=int(slips.any(1).sum()),
                        tight_joint=result['joint_names'][joints[tight]] if flags[stop,i,4] else None,
                        tight_joint_margin_rad=round(float(joint_margin[tight]),5) if flags[stop,i,4] else None,
                        trace_sha256=audit['files']['trajectory.npz'])
            reports.append(skew)
            if name=='Armchair' and scale==88:
                plot_example((np.arange(stop+1)+1)*DT,over,seat,foot_forces,skew['stop_s'])
    return reports


def main():
    calibration=read(MODEL)
    protocol = read(CATALOG/'protocol.json')
    cases = []
    for scale in SIZES:
        folder = CATALOG/f'piecewise_{scale}'
        cases.extend(diagnose(folder,protocol['roster'],scale,calibration))
    write(STUDY/'original_forward_balance.json',dict(calibration_sha256=sha(MODEL),cases=cases))
    with (STUDY/'original_forward_balance.csv').open('w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    for c in cases:
        if c['asset'] in ('Armchair','Hazelwood_Chair','Bevo','Dutchtown_Sofa','MapleBarStool','Petite_Chair'):
            print({k:v for k,v in c.items() if k!='trace_sha256'},flush=True)


if __name__ == '__main__':
    main()
