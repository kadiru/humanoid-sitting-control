"""Sensor-feedback PPO with complete, undiscounted episode reward-to-go.

The previous three applied residuals are explicit controller memory. There is
no short-horizon bootstrap, and no contact/pose ground truth in network inputs.
"""
import torch
from torch import nn
from posture_episode_ppo import PosturePPO


class FeedbackPPO(PosturePPO):
    @staticmethod
    def network(n):
        return nn.Sequential(nn.Linear(417,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,n))

    def state(self):
        result=super().state()
        result['contract']='414 sensor/reference channels + 3 previous normalized residuals; repeated symmetric hip/knee/ankle actions; undiscounted complete-episode reward-to-go'
        return result


def slew(previous,target,dt=.1,rate=.004,span=.004):
    """Bound velocity as well as absolute learned residual; zero is identity."""
    target=target.clamp(-span,span)
    return (previous+(target-previous).clamp(-rate*dt,rate*dt)).clamp(-span,span)


def complete_returns(torque_integrals,baseline_integrals,terminal_penalty,decision_steps,scale=1350.):
    """Each entry covers five physical ticks (0.1 s), in Nm² seconds.

    Baseline is action-independent, from a matched development reference only.
    A terminal penalty is charged to all decisions in that full trajectory.
    """
    assert torque_integrals.shape==baseline_integrals.shape
    assert torque_integrals.ndim==2 and len(torque_integrals)==200
    per_step=-(torque_integrals-baseline_integrals)/scale
    tail=per_step.flip(0).cumsum(0).flip(0)
    return tail[decision_steps]-terminal_penalty[None,:]
