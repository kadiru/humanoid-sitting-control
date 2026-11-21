"""Summarize supported rest, contact severity and arm position from scored trajectories."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from compare_touchdown_ppo import verify_audit
from evaluate_chair_contact_motion import signatures
from seated_arm_posture import BASE
from study_live_sitting import read, write

CHANNELS = ['pelvis', 'left thigh', 'right thigh', 'torso', 'back cover', 'left foot', 'right foot']


def describe(run):
    p = run/'evaluation'
    audit, rows = verify_audit(p)
    names = read(p/'manifest.json')['metadata']['body_names']
    with np.load(p/'trajectory.npz') as z, np.load(p/'contact_attribution.npz') as probes:
        b,slip,force = z['body'],z['physical_slip_m_s'],z['physical_force_w']
        speed = np.where(np.linalg.norm(force,axis=-1)>.01*z['weight'][...,None],slip,0.)
        exceeded = speed>.02
        np.testing.assert_array_equal(exceeded.any(-1),~z['no_slip'])
        torso = b[:,:,names.index('torso_link')]
        palms = b[:,:,[names.index(s+'_palm_link') for s in ('left','right')],:3]
        rot = Rotation.from_quat(torso[...,3:].reshape(-1,4)).as_matrix().reshape(*torso.shape[:2],3,3)
        local = np.einsum('tnhj,tnjk->tnhk',palms-torso[:,:,None,:3],rot)
        span = abs(local[:,:,:,1].max(-1)-local[:,:,:,1].min(-1))
        # Descriptive arm measurement for the final five seconds, independent of the pass label.
        in_front = (local[...,0]>0).all(-1) & (abs(local[...,1])<.30).all(-1)
        wrong = z['wrong_force']
        flags = z['flags']
        inside = z['physical_inside_patch']
        load = z['support_bw']
        weight = z['weight']
        probe_force = probes['chair_history_peak_N']
        probe_names = probes['probe_links'].tolist()
        net = probes['net_history_peak_N']
        unlisted = [i for i,n in enumerate(probes['body_names'].tolist()) if n not in probes['filtered_body_names']]
        all_names = probes['body_names'].tolist()
        data=[]
        for i,row in enumerate(rows):
            t=np.flatnonzero(flags[:,i,1])
            first=None
            if len(t):
                k=int(t[0])
                external = int(np.argmax(net[k,i,unlisted]))
                filtered = int(probe_force[k,i].argmax())
                outside = (np.linalg.norm(force[k,i],axis=-1)>.01*weight[k,i]) & ~inside[k,i]
                first=dict(tick=k, wrong_force_N=float(wrong[k,i]), outside=[CHANNELS[n] for n in np.flatnonzero(outside)],
                           strongest_unlisted_link=all_names[unlisted[external]] if wrong[k,i]>10 else None,
                           strongest_chair_probe=probe_names[filtered] if probe_force[k,i,filtered]>0 else None,
                           chair_probe_N=float(probe_force[k,i,filtered]))
            travel=np.where(exceeded[:,i],speed[:,i],0.).sum(0)*20  # m/s x .02 s x 1000 mm/m
            data.append(dict(id=row['id'],clean=row['clean'],stable=row['stable'],hold_s=row['final_hold_s'],
                         slips=row['slip_ticks'],illegal_ticks=row['illegal_ticks'],failures=row['physical_failures'],
                         estimated_travel_while_over_mm=dict(zip(CHANNELS,travel.tolist())),
                         slip_peak_m_s=float(speed[:,i].max()),first_forbidden=first,
                         palms_forward_fraction=float(in_front[-250:,i].mean()),
                         hand_span_m=float(np.median(span[-250:,i])),
                         palm_x_m=float(np.median(local[-250:,i,:,0])),
                         rest_load_bw=float(load[-250:,i].mean()),
                         mechanical_power_W=audit['final_absolute_mechanical_power_W'][i]))
    return data


def qualifies(old,new):
    return (new['clean'] and new['palms_forward_fraction']>=.95 and
            old['palms_forward_fraction']<.95 and all(a>=b for a,b in zip(old['failures'],new['failures'])))


def report(out, run, profile=None):
    a,b=describe(BASE),describe(run)
    assert [r['id'] for r in a]==[r['id'] for r in b]
    pairs=list(zip(a,b))
    def counts(rows):
        return dict(clean=sum(r['clean'] for r in rows),supported_five_seconds=sum(r['stable'] for r in rows),
                    clean_with_hands_forward=sum(r['clean'] and r['palms_forward_fraction']>=.95 for r in rows),
                    hands_forward=sum(r['palms_forward_fraction']>=.95 for r in rows))
    qualified=[i for i,(old,new) in enumerate(pairs) if qualifies(old,new)]
    result=dict(before=counts(a),after=counts(b),qualified=qualified,
                strict_lost=[old['id'] for old,new in pairs if old['clean'] and not new['clean']],
                strict_gained=[old['id'] for old,new in pairs if new['clean'] and not old['clean']],
                cases=[dict(before=old,after=new) for old,new in pairs])
    write(out/'arm_and_contact_audit.json',result)
    if profile and qualified:
        scans=np.array(signatures(BASE))[qualified]
        sites=np.where(np.isfinite(scans),scans.astype(object),None).tolist()
        write(out/'tested_profile.json',dict(read(profile),sites=sites))
    print({k:v for k,v in result.items() if k!='cases'},flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--profile',type=Path)
    a=p.parse_args()
    report(a.run,a.run,a.profile)
