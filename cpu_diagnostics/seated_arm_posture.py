"""Pilot lap posture with the original sitting controller, filters, chair contacts and score."""
import argparse
from pathlib import Path
import sys

import numpy as np

BASE = Path('/dev/shm/right_hip_repair_final/matched')
HAND_SIDES = ('left', 'right')
ARM_JOINTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow_pitch', 'elbow_roll')


def install(launcher, options, profile_path):
    sys.path[:0] = [str(launcher.parent), str(launcher.parent/'trainer')]
    import torch
    from chair_contact_motion import ChairContactMotion
    from evaluate_armchair_pair import trace_contacts
    from study_live_sitting import read, write
    from chair_contact_motion import distances, descriptor

    profile = read(profile_path)
    nominal_motion = ChairContactMotion.apply

    def apply(motion, learner, obs, tick, nominal):
        env = learner.env
        latent = nominal_motion(motion,learner,obs,tick,nominal)
        start = profile['start_tick']
        if tick == start:
            joint_names = env.metadata['joint_names']
            indices = [joint_names.index(f'{side}_{joint}_joint') for side in HAND_SIDES for joint in ARM_JOINTS]
            env.arm_indices = torch.tensor(indices, device=latent.device)
            env.arm_anchor = latent[:,indices].clone()
            # At this time the outgoing and incoming shoulder-roll commands in the
            # original motion library are already separated, after the rear scan.
            active = env.arm_anchor[:,1] > profile['outgoing_roll_rad']
            if 'sites' in profile:
                rims, planes, tiles = map(np.asarray,zip(*motion.measurements,strict=True))
                masks = []
                for i in range(len(active)):
                    scan = descriptor(rims[:,i],planes[:,i],tiles[:,i])
                    masks.append(distances(scan,np.array(profile['sites'],float)).min(initial=np.inf)<.50)
                active &= torch.tensor(masks,device=latent.device)
            env.arm_active = active
            env.arm_goal = torch.tensor(profile['target_rad'],dtype=latent.dtype,device=latent.device)
            write(env.out/'arm_assignment.json',dict(tick=tick,active=active.tolist(),joint_indices=indices))
        if tick >= start:
            elapsed = np.clip((tick-start)/profile['duration_ticks'],0.,1.)
            blend = float(elapsed**2*(3.-2.*elapsed))
            arm = torch.lerp(env.arm_anchor,env.arm_goal,blend)
            latent = latent.clone()
            latent[:,env.arm_indices] = torch.where(env.arm_active[:,None],arm,latent[:,env.arm_indices])
        return latent
    ChairContactMotion.apply = apply
    trace_contacts(launcher,options)


def evaluate(out, profile):
    from evaluate_standing_rim import comparison
    from evaluate_seat_plane_approach import DISK_RESERVE
    from study_live_sitting import read, sha, write, stage, PREPARED

    f = read(BASE/'evaluation.freeze.json')
    previous = f['command']
    inputs = [Path(k) for k in f['inputs']]
    assert all(sha(p)==f['inputs'][str(p)] for p in inputs)
    bank = Path(previous[previous.index('--bank')+1])
    options = previous[3:previous.index('--mode')]
    assert options[0]=='--worker'
    out.mkdir(exist_ok=True)
    write(out/'reservation.json',dict(files={str(p):sha(p) for p in bank.iterdir()}))
    stage(out,'evaluation',bank,PREPARED/'library.npz',worker=Path(__file__).resolve(),
          extra=['--arm-worker',str(profile),*options[1:]],
          inputs=[*inputs, BASE/'evaluation.freeze.json', profile, Path(__file__).with_name('evaluate_armchair_pair.py')],
          disk_reserve=DISK_RESERVE)
    score=comparison(out,BASE)
    write(out/'outcome.json',score)
    print(score,flush=True)


if __name__=='__main__':
    if sys.argv[1]=='--arm-worker':
        install(Path(sys.argv[3]),sys.argv[4:],Path(sys.argv[2]))
    else:
        p=argparse.ArgumentParser(description=__doc__)
        p.add_argument('out',type=Path)
        p.add_argument('profile',type=Path)
        args=p.parse_args()
        evaluate(args.out.resolve(),args.profile.resolve())
