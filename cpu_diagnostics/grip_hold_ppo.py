"""Training credit and a hold handoff based on measured seat contact forces."""

import copy
from pathlib import Path
import sys

import torch


HOLD_TICK = 750
HISTORY = 50
RAMP = 50
ENTRY_UTILIZATION = .45
RETURN_UTILIZATION = .65
GRIP_LIMIT = .90
MIN_LOAD = .05


def seat_state(metrics):
    force = metrics['force_w'][:, :5]
    normal = metrics['normal_w'][:, :5]
    normal_force = (force*normal).sum(-1)
    tangent = (force-normal_force[...,None]*normal).norm(dim=-1)
    weight = metrics['bodyweight_N']
    active = force.norm(dim=-1)>.01*weight[:,None]
    capacity = (metrics['friction'][:, :5]*normal_force.clamp_min(0)).clamp_min(1e-7)
    utilization = torch.where(active & (tangent>1e-7),tangent/capacity,0.).amax(1)
    support = (normal_force.clamp_min(0)*metrics['rest_patch'][:,:5]*
               metrics['permitted'][:,:5]).sum(-1)/weight
    return utilization,support


def hold_credit(utilization,load,power,good):
    """Dense credit for grip reserve and lower mechanical power at safe support."""
    supported = good & (load>=MIN_LOAD)
    grip = .18*(1.-utilization/GRIP_LIMIT).clamp(0,1)*supported
    effort = .12*.50/(.50+power)*supported
    risk = .90*((utilization-RETURN_UTILIZATION)/(GRIP_LIMIT-RETURN_UTILIZATION)).clamp(0,2).square()
    return grip,effort,risk


def capture_metrics(support):
    original = support.metrics
    state = {}

    def metrics(raw):
        result = original(raw)
        state['metrics'] = result[0]
        return result

    support.metrics = metrics
    return state


def train_worker():
    here = Path(__file__).resolve().parent
    sys.path.insert(0,str(here/'trainer'))
    import run_training
    import run_supported_postures

    state = capture_metrics(run_supported_postures)
    environment = run_training.FullBodyEnv

    class HoldRewardEnv(environment):
        def step(self,*args,**kwargs):
            obs,reward,done,learn,horizon = super().step(*args,**kwargs)
            if self.t<=HOLD_TICK:
                return obs,reward,done,learn,horizon
            metrics = state['metrics']
            utilization,load = seat_state(metrics)
            power = (metrics['torque']*metrics['joint_velocity']).abs().sum(-1)
            grip,effort,risk = hold_credit(utilization,load,power,self.held>0)
            if horizon:
                clean = self.alive & (self.held>=250) & (self.slip_ticks==0) & (self.illegal_ticks==0)
                original_effort = 40*clean/(1+10*self.effort_sum/self.effort_ticks.clamp_min(1))
                reward = reward-original_effort
            return obs,reward+torch.where(learn,grip+effort-risk,0.),done,learn,horizon

    run_training.FullBodyEnv = HoldRewardEnv
    run_training.main()


class GripHandoff:
    def __init__(self,actor,trained):
        self.actor = actor
        self.trained = trained
        self.peak = None
        self.enabled = None
        self.alpha = None
        self.ticks = None

    def observe(self,tick,metrics):
        if tick<=HOLD_TICK-HISTORY:
            return
        utilization,load = seat_state(metrics)
        if tick<=HOLD_TICK:
            self.peak = utilization if self.peak is None else torch.maximum(self.peak,utilization)
        if tick==HOLD_TICK:
            self.enabled = (self.peak<ENTRY_UTILIZATION) & (load>=MIN_LOAD)
            self.entry = self.enabled.clone()
            self.ticks = torch.zeros_like(utilization,dtype=torch.int32)
        if tick>=HOLD_TICK:
            self.enabled &= (utilization<RETURN_UTILIZATION) & (load>=MIN_LOAD) & ~metrics['failure']
            phase = min((tick-HOLD_TICK)/RAMP,1.)
            self.alpha = self.enabled[:,None]*(phase*phase*(3-2*phase))
            self.ticks += (self.alpha[:,0]>0)

    def __call__(self,obs):
        original = self.actor(obs)
        if self.alpha is None:
            return original
        return original+self.alpha*(self.trained(obs)-original)

    def receipt(self):
        return dict(entry_peak_utilization=self.peak.cpu().tolist(),entry_eligible=self.entry.cpu().tolist(),
                    active_ticks=self.ticks.cpu().tolist(),enabled_at_end=self.enabled.cpu().tolist())


def evaluate_worker(checkpoint):
    from live_sitting_edge import worker
    from repair_live_touchdown import HERE, ROOT, sha, write
    sys.path[:0] = [str(HERE/'trainer'),str(ROOT/'scripts')]
    import run_supported_postures
    from policy import Learner

    state = capture_metrics(run_supported_postures)
    trained = torch.load(checkpoint,map_location='cpu',weights_only=False)
    original_load = Learner.load
    original_choose = Learner.choose

    def load(self,path):
        baseline = original_load(self,path)
        for key in ('mean','scale'):
            torch.testing.assert_close(baseline[key].cpu(),trained[key],rtol=0,atol=0)
        assert int(trained['sitting']['hold_start_tick'])==int(self.sitting['hold_start_tick'])==HOLD_TICK
        actor = copy.deepcopy(self.actor)
        actor.load_state_dict(trained['actor'])
        self.actor = GripHandoff(self.actor,actor)
        return baseline

    def choose(self,obs,stochastic=True):
        assert not stochastic
        tick = round(float(obs[0,335])*1000)
        if tick>HOLD_TICK-HISTORY:
            self.actor.observe(tick,state['metrics'])
        if tick==999:
            write(self.env.out/'ppo_actor.json',dict(first_tick=HOLD_TICK,checkpoint_sha256=sha(checkpoint)))
            write(self.env.out/'grip_handoff.json',self.actor.receipt())
        return original_choose(self,obs,stochastic)

    Learner.load = load
    Learner.choose = choose
    worker()


if __name__=='__main__':
    mode = sys.argv.pop(1)
    if mode=='train':
        train_worker()
    else:
        assert mode=='evaluate'
        evaluate_worker(Path(sys.argv.pop(1)))
