"""Bounded control options shared by separate sitting skills and one sequenced actor."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from rescan_sitting import shoe_frame, smooth

ALIGN, DESCEND, TRANSFER, REST = range(4)
WIDTH = 16
ACTIONS = 4


class Options(nn.Module):
    def __init__(self, separate):
        super().__init__()
        self.separate = separate
        width = 20 if separate else 48
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(WIDTH, width), nn.Tanh(), nn.Linear(width, width),
                          nn.Tanh(), nn.Linear(width, ACTIONS))
            for _ in range(4 if separate else 1)
        ])

    def forward(self, x, phase):
        if not self.separate:
            return self.heads[0](x)
        return torch.stack([h(x) for h in self.heads], 1)[torch.arange(len(x), device=x.device), phase]


class Direct(nn.Module):
    """One network runs from the rear scan to the last 50 Hz joint command."""
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.net = nn.Sequential(nn.Linear(448,96),nn.ELU(),nn.Linear(96,96),nn.ELU(),nn.Linear(96,23))

    def forward(self, obs):
        x = ((obs-self.mean)/self.scale).clamp(-8.,8.)
        return self.net(torch.cat((x,torch.ones_like(x[:,:1])),1))

    def align(self, features):
        x = torch.zeros((len(features),448),device=features.device)
        x[:,:WIDTH] = features
        return self.net(x)[:,:ACTIONS]


class Decisions:
    def __init__(self, config):
        self.config = config
        self.mode = config['mode']
        self.device = torch.device('cuda:0')
        self.network = None
        self.direct = None
        if self.mode == 'direct':
            state = torch.load(config['weights'], map_location=self.device, weights_only=True)
            self.direct = Direct(state['model']['mean'],state['model']['scale']).to(self.device)
            self.direct.load_state_dict(state['model'])
            self.direct.eval()
        if self.mode in ('sample', 'evaluate'):
            state = torch.load(config['weights'], map_location=self.device, weights_only=True)
            self.network = Options(state['separate']).to(self.device)
            self.network.load_state_dict(state['model'])
            self.network.eval()
        self.rng = torch.Generator(device=self.device).manual_seed(config['seed'])
        self.records = {}

    @torch.no_grad()
    def choose(self, row, phase, values, enabled=True):
        key = (row, phase)
        if key in self.records:
            return self.records[key][1]
        vector = np.zeros(WIDTH, np.float32)
        vector[phase] = 1.
        vector[4:4+len(values)] = values
        verified = self.config.get('verified_rows')
        used = enabled and phase in self.config['phases'] and (verified is None or row in verified)
        action, logp = 0, 0.
        if used and self.mode == 'probe':
            action = self.config['choices'][row][phase]
        if used and self.network is not None:
            x = torch.as_tensor(vector, device=self.device)[None]
            logits = self.network(x, torch.tensor([phase], device=self.device))[0]
            dist = logits.log_softmax(0)
            if self.mode == 'sample':
                action = int(torch.multinomial(dist.exp(), 1, generator=self.rng))
            else:
                action = int(logits.argmax())
            logp = float(dist[action])
        if used and self.direct is not None:
            x = torch.as_tensor(vector,device=self.device)[None]
            distribution = self.direct.align(x)[0].log_softmax(0)
            action = int(distribution.argmax())
            logp = float(distribution[action])
        self.records[key] = vector, action, logp, used
        return action

    def save(self, path):
        keys = list(self.records)
        np.savez_compressed(path, row=[k[0] for k in keys], phase=[k[1] for k in keys],
                            features=np.stack([self.records[k][0] for k in keys]),
                            action=[self.records[k][1] for k in keys],
                            logp=[self.records[k][2] for k in keys],
                            enabled=[self.records[k][3] for k in keys])

    def snapshot(self, env):
        data = env.robot.data
        bw = data.body_mass.torch.sum(1).cpu().numpy() * 9.81
        seat_sensor = env.raw.scene['pelvis_seat'].data
        normal = seat_sensor.normal_force_matrix_w.torch.sum((1,2)).cpu().numpy()
        tangential = seat_sensor.friction_force_matrix_w.torch.sum((1,2)).cpu().numpy()
        left, right = (env.raw.scene['rest_' + side + '_floor'].data.normal_force_matrix_w.torch
                       .sum((1,2))[:,2].cpu().numpy()/bw for side in ('left','right'))
        up = data.projected_gravity_b.torch.cpu().numpy()
        omega = data.root_ang_vel_b.torch.norm(dim=1).cpu().numpy()
        speed = data.root_lin_vel_w.torch.norm(dim=1).cpu().numpy()
        q = data.joint_pos.torch.cpu().numpy()
        vel = data.joint_vel.torch[:,:23]
        torque = data.applied_torque.torch[:,:23]
        power = (vel * torque).abs().cpu().numpy()
        rpm = power[:,[15,16,19,20]].sum(1)
        hold_power = power.sum(1)
        load = normal[:,2]/bw
        shear = np.linalg.norm(tangential[:,:2],axis=1) / np.maximum(np.linalg.norm(normal,axis=1),1.)
        return tuple(zip(left,right,up,omega,speed,q,rpm,hold_power,load,shear,strict=True))

    def measure(self, snapshot, row):
        left,right,up,omega,speed,q,rpm,hold_power,load,shear = snapshot[row]
        # Camera-relative alignment is retained alongside the current proprioception.
        align, *_ = self.records.get((row, ALIGN), (np.zeros(WIDTH),))
        features = [*align[4:7], load, (left-right), min(left,right), *up[:2], omega*2,
                    np.clip(rpm*.1, 0., 3.), np.clip(hold_power*.1, 0., 3.), shear]
        return features, load, min(left,right), omega, speed, q, up

    def attach(self, downstream):
        from chair_contact_motion import ChairContactMotion
        previous = ChairContactMotion.apply
        config = self.config

        def apply(motion, learner, obs, tick, nominal):
            if self.direct is not None:
                if tick == 999:
                    self.save(learner.env.out/'skill_decisions.npz')
                with torch.no_grad():
                    return self.direct(obs)
            out = previous(motion, learner, obs, tick, nominal)
            env = learner.env
            if tick == 0:
                motion.learned_stages = [dict(contact=0, stable=0, targets={}, last=None)
                                         for _ in range(len(obs))]
                motion.learned_entry = {}
            has_options = any(p in config['phases'] for p in (DESCEND, TRANSFER, REST))
            if tick >= 160 and has_options and (config['phases'] != [REST] or tick >= 680):
                snapshot = self.snapshot(env)
                live = env.alive.cpu().numpy()
            else:
                snapshot,live = (),()
            for row in np.flatnonzero(live):
                phase_state = motion.learned_stages[row]
                targeted_hold = row in config['hold_rows']
                if config['phases'] == [REST] and (not targeted_hold or tick < 680):
                    continue
                features, seat, foot, omega, speed, joints, grav = self.measure(snapshot, row)
                targets = phase_state['targets']
                if tick == 160 and DESCEND in config['phases']:
                    a = self.choose(row, DESCEND, features, foot > .15 and speed < .045)
                    phase_state['last'] = out[row].clone()
                    targets[DESCEND] = (a, tick)
                if DESCEND in targets:
                    a, start = targets[DESCEND]
                    if a and tick < 330 and seat < .025:
                        # Slow early command changes and fade back to the same
                        # fixed-foot approach before the contact handoff.
                        weight = [.0, .22, .38, .53][a]*smooth((tick-start)/28.)*(1-smooth((tick-283)/47.))
                        out[row] = (1-weight)*out[row]+weight*phase_state['last']
                        phase_state['last'] = out[row].clone()
                if TRANSFER in config['phases'] and TRANSFER not in targets and tick >= 340:
                    phase_state['contact'] = phase_state['contact']+1 if seat>.075 and foot>.16 else 0
                    if phase_state['contact'] >= 8:
                        targets[TRANSFER] = (self.choose(row, TRANSFER, features),tick)
                if REST in config['phases'] and REST not in targets and tick >= 680:
                    phase_state['stable'] = phase_state['stable']+1 if seat>.09 and foot>.12 and omega<.15 and speed<.035 else 0
                    if phase_state['stable'] >= 5:
                        targets[REST] = (self.choose(row, REST, features, targeted_hold),tick)
                        motion.learned_entry[row] = tick
                if TRANSFER in targets and REST not in targets:
                    a, start = targets[TRANSFER]
                    weight = smooth((tick-start)/48.)*(1-smooth((tick-start-70)/45.))
                    if a and weight:
                        adjust(out, row, env.model, joints, grav, weight,
                               [0.,-.008,-.015,.008][a], [0.,.00035,.00075,.0003][a])
                if REST in targets:
                    a, start = targets[REST]
                    if a:
                        weight = smooth((tick-start)/64.)
                        # Action three also removes a small asymmetric moment
                        # by following the more heavily loaded foot's roll.
                        roll = .014*np.sign(features[4]) if a == 3 else 0.
                        adjust(out, row, env.model, joints, grav, weight,
                               [0.,-.028,-.052,-.028][a], [0.,.0012,.0017,.0012][a], roll)
            if tick == 999:
                self.save(env.out/'skill_decisions.npz')
            return out

        ChairContactMotion.apply = apply
        return downstream()


def adjust(out, row, model, q, gravity, weight, pitch, drop, roll=0.):
    from stance23_planner import chains, legs, solve

    feet, _, basis = shoe_frame(model, q, gravity)
    turn = basis @ Rotation.from_euler('xy', weight*np.array([roll,pitch])).as_matrix() @ basis.T
    pivot = np.array([-.021,0.,-.147])
    move = pivot-turn@pivot-basis[:,2]*(weight*drop)
    for i,(chain,ids) in enumerate(zip(chains,legs)):
        goal = np.eye(4)
        goal[:3,:3] = turn.T@feet[i,:3,:3]
        goal[:3,3] = turn.T@(feet[i,:3,3]-move)
        target,error = solve(chain,q[ids].copy(),goal,ids)
        if error<.001:
            out[row,ids] += out.new_tensor(np.clip(target-q[ids],-.095,.095))


def attach_wire_template(config, downstream):
    """Allow a standing alignment choice to initialize the same rim feedback."""
    from chair_contact_motion import ChairContactMotion

    previous = ChairContactMotion.apply
    assets = config['templates_by_row']
    prototype = assets.index('Chair_Wire')

    def apply(motion, learner, obs, tick, nominal):
        out = previous(motion, learner, obs, tick, nominal)
        if tick == 0 and not any(assets[i] == 'Chair_Wire' for i in motion.rescan_feedback):
            # Planted Wire paths copy a neutral instance of the same feedback
            # state. It is a template only, with no associated motion path.
            motion.rescan_feedback[prototype] = dict(
                error=0., velocity=0., filtered=np.zeros(2), clock=0., enabled=False,
                recenter=True, anchor=None, seat_trim=0., last_x=None, x_speed=0.,
                settled=0, hold_target=None, log=[])
        return out

    ChairContactMotion.apply = apply
    return downstream()
