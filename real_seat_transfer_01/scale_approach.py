"""A one-decision sitting policy. All approach inputs come from the standing scan."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn

LEG = [0,1,3,4,7,8,11,12,15,16,19,20]
N_ACTIONS = 5


def edge_features(height, direct, frames, model):
    from seat_plane_approach import stance_axes
    direct = np.asarray(direct)
    direct = direct[np.isfinite(direct)]
    if len(direct) >= 2 and height > 0:
        return [height, -float(np.median(direct))], 'rim'
    estimates = []
    for depth, intrinsics, q, gravity, mount in frames:
        poses = model.fk(q)
        feet = np.stack([poses[s+'_ankle_roll_link'] for s in ('left','right')])
        axes = stance_axes(feet, gravity)
        up = axes[:,2]
        origin = feet[:,:3,3].mean(0)
        origin += up*np.mean([(model.vertices[s+'_ankle_roll_link'] @ f[:3,:3].T @ up).min()
                              for s,f in zip(('left','right'),feet)])
        v,u = np.indices(depth.shape)
        valid = np.isfinite(depth) & (depth>.07) & (depth<1.8)
        z = np.where(valid,depth,0)
        points = np.stack([(u-intrinsics[0,2])*z/intrinsics[0,0],
                           (v-intrinsics[1,2])*z/intrinsics[1,1],z],-1) @ mount[:3,:3].T + mount[:3,3]
        xyz = (points-origin) @ axes
        dx = xyz[1:-1,2:]-xyz[1:-1,:-2]
        dy = xyz[2:,1:-1]-xyz[:-2,1:-1]
        n = np.cross(dx,dy)
        x,y,h = xyz[1:-1,1:-1].transpose(2,0,1)
        keep = (valid[1:-1,1:-1] & (abs(h-height)<.035) & (abs(y)<.09) & (x<-.075) & (x>-.60)
                & (abs(n[:,:,2])>.84*np.linalg.norm(n,axis=-1))
                & (np.linalg.norm(dx,axis=-1)<.04) & (np.linalg.norm(dy,axis=-1)<.04))
        if keep.sum() >= 12:
            outside = model.self_mask(points[1:-1,1:-1][keep], poses, margin=.003)
            if outside.sum() >= 12:
                estimates.append(-np.percentile(x[keep][outside],96))
    if len(estimates)>=2 and height>0:
        return [height,float(np.median(estimates))], 'visible_patch'
    return [height,0.], 'unavailable'


class Approach(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.actor = nn.Linear(32,N_ACTIONS)
        self.value = nn.Linear(32,1)

    def forward(self, features):
        x = (features-features.new_tensor([.39,.22]))/features.new_tensor([.095,.07])
        h = self.hidden(x.clamp(-4,4))
        return self.actor(h),self.value(h).flatten()


def path_for(features, choice, library):
    # Continuous geometry selects a measured trajectory. The policy selects how
    # much of a fixed-foot inverse-kinematics correction to apply before loading.
    keys = library['features']
    delta = features[:,None,:]-keys[None,:,:]
    distance = (delta/np.array([.040,.065]))**2
    donor = np.argmin(distance.sum(2),axis=1)
    delta = features-keys[donor]
    dz = np.clip(delta[:,0],-.085,.085)
    dx = np.clip(-delta[:,1],-.06,.06)
    # Action 0 uses the incumbent. 1 reproduces the donor, 2 corrects height,
    # 3 corrects both rearward reach and height.
    dz = dz*(choice>=2)
    dx = dx*(choice>=3)
    correction = np.clip(library['response'][donor,:,:,0]*dx[:,None,None]
                         +library['response'][donor,:,:,1]*dz[:,None,None],-.24,.24)
    return (library['desired'][donor]+correction).astype('f'), donor


def attach_approach(launcher, options, attach):
    from chair_contact_motion import ChairContactMotion
    prior = ChairContactMotion.apply
    ev = Path(options[options.index('--output')+1])
    config = json.loads((ev.parent/'approach.json').read_text())
    def apply(motion, learner, obs, tick, nominal):
        before = prior(motion,learner,obs,tick,nominal)
        if tick < 40:
            return before
        if tick == 40:
            env = learner.env
            seen = json.loads((ev/'seen_seat.json').read_text())
            matched = np.array([r[4] for r in json.loads((ev/'camera_method.json').read_text())])
            rims = np.asarray([s[0] for s in motion.measurements])[:,:,0]
            features, modes = [], []
            for i,h in enumerate(seen['estimated_height_m']):
                frames = [(s[1][i],s[2][i],s[3][i],s[4][i],s[8]) for s in env.standing_samples]
                f,m = edge_features(h, rims[:,i], frames, env.model)
                features.append(f)
                modes.append(m)
            features = np.array(features)
            active = (~matched) & (np.array(modes)!='unavailable')
            logp = np.zeros(len(obs))
            values = np.zeros(len(obs))
            if config['mode']=='collect':
                choice = np.array(config['choices'],dtype=int)
            else:
                network = Approach().to(before.device)
                network.load_state_dict(torch.load(config['checkpoint'],map_location=before.device,weights_only=True))
                ft = torch.as_tensor(features,dtype=before.dtype,device=before.device)
                logits,val = network(ft)
                dist = torch.distributions.Categorical(logits=logits)
                if config['mode']=='sample':
                    rng = torch.Generator(device=before.device).manual_seed(config['seed'])
                    actions = torch.multinomial(dist.probs,1,generator=rng).flatten()
                else:
                    actions = logits.argmax(dim=1)
                choice = actions.cpu().numpy()
                logp = dist.log_prob(actions).cpu().numpy()
                values = val.cpu().numpy()
            with np.load(config['library']) as lib:
                path,donors = path_for(features,choice,lib)
            motion.scale_leg_path = before.new_tensor(path)
            motion.scale_active = torch.as_tensor(active & (choice!=0),device=before.device)
            motion.fold_on_approach = torch.as_tensor(active & (choice==4),device=before.device)
            arms = json.loads(Path(__file__).with_name('folded_arms.json').read_text())['arms']
            motion.fold_indices = arms['left']['joints']+arms['right']['joints']
            motion.fold_angles = before.new_tensor(arms['left']['angles']+arms['right']['angles'])
            np.savez(ev/'approach_choice.npz',features=features,edge_source=modes,active=active,
                     choice=choice,old_logp=logp,old_value=values,donor=donors)
        target = before.clone()
        weight = min(1.,(tick-40)/55.)
        blended = torch.lerp(before[:,LEG],motion.scale_leg_path[:,tick],weight)
        target[:,LEG] = torch.where(motion.scale_active[:,None],blended,before[:,LEG])
        # Use the existing handoff clock, which starts on measured seat support.
        # This option keeps the hands clear while the legs lower the pelvis.
        fold_weight = min(1.,(tick-40)/75.)*(1.-motion.arm_handoff)
        arms = motion.fold_indices
        folded = torch.lerp(before[:,arms],motion.fold_angles.expand(len(obs),-1),fold_weight[:,None])
        target[:,arms] = torch.where(motion.fold_on_approach[:,None],folded,before[:,arms])
        return target
    ChairContactMotion.apply = apply
    attach(launcher,options)
