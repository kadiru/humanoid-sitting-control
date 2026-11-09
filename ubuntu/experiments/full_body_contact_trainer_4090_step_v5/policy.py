"""Independent23-channel PPO with bounded joint-target increments and GAE."""
import copy
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from common import OBS, STEP, FILTER, MARGIN


def command(previous, latent, lower, upper):
    desired = (previous+(STEP/FILTER)*torch.tanh(latent)).clamp(lower+MARGIN, upper-MARGIN)
    delta = (FILTER*(desired-previous)).clamp(-STEP, STEP)
    return previous+delta


def advantages(reward, value, next_value, done, gamma=.999, lam=.97):
    # done includes physical termination AND finite-horizon task completion.
    result = torch.zeros_like(reward)
    carry = torch.zeros_like(reward[0])
    for t in range(len(reward)-1, -1, -1):
        continuation = (~done[t]).float()
        delta = reward[t]+gamma*next_value[t]*continuation-value[t]
        carry = delta+gamma*lam*continuation*carry
        result[t] = carry
    return result, result+value


class Learner:
    def __init__(self, mean, scale, seed, device='cpu'):
        torch.manual_seed(seed)
        self.device = device
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.scale = torch.as_tensor(scale, dtype=torch.float32, device=device).clamp_min(.05)
        assert self.mean.shape == self.scale.shape == (OBS,)
        def net(outputs):
            return nn.Sequential(nn.Linear(OBS, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, outputs)).to(device)
        self.actor = net(23)
        self.critic = net(1)
        self.actor[-1].weight.data.zero_(); self.actor[-1].bias.data.zero_()
        self.std = .25
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.updates = 0
        self.actor_steps = 0
        self.critic_steps = 0

    def normalize(self, obs):
        return ((obs-self.mean)/self.scale).clamp(-10, 10)

    @torch.no_grad()
    def choose(self, obs, stochastic=True):
        x = self.normalize(obs)
        mu = self.actor(x)
        dist = Normal(mu, self.std)
        action = dist.sample() if stochastic else mu
        return action, dist.log_prob(action).sum(-1), self.critic(x).squeeze(-1), mu

    def update(self, obs, action, logp, old_mean, advantage, target):
        x = self.normalize(obs)
        adv = (advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-6)
        accepted = rejected = candidate_tests = 0
        accepted_scales = []
        for epoch in range(4):
            # Whole-batch KL rejection restores both parameters AND Adam state.
            before = copy.deepcopy(self.actor.state_dict())
            opt_before = copy.deepcopy(self.actor_opt.state_dict())
            dist = Normal(self.actor(x), self.std)
            ratio = (dist.log_prob(action).sum(-1)-logp).exp()
            loss = -torch.minimum(ratio*adv, ratio.clamp(.8, 1.2)*adv).mean()
            self.actor_opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
            self.actor_opt.step()
            proposed = copy.deepcopy(self.actor.state_dict())
            admitted = False
            initial_loss = float(loss.detach())
            with torch.no_grad():
                # One Adam proposal, at most eight bounded candidate evaluations.
                # Adam moments count the observed gradient once, regardless of
                # accepted step length. Rejected proposals restore all state.
                for exponent in range(8):
                    alpha = 2.**(-exponent)
                    if exponent:
                        self.actor.load_state_dict({k: before[k]+alpha*(proposed[k]-before[k]) for k in before})
                    mean = self.actor(x)
                    kl = ((mean-old_mean).square()/(2*self.std**2)).sum(-1).mean()
                    candidate_ratio = (Normal(mean,self.std).log_prob(action).sum(-1)-logp).exp()
                    candidate_loss = -torch.minimum(candidate_ratio*adv,candidate_ratio.clamp(.8,1.2)*adv).mean()
                    candidate_tests += 1
                    if torch.isfinite(kl) and torch.isfinite(candidate_loss) and kl <= .02 and candidate_loss <= initial_loss:
                        admitted = True
                        accepted_scales.append(alpha)
                        break
            if not admitted:
                self.actor.load_state_dict(before); self.actor_opt.load_state_dict(opt_before)
                rejected += 1
                break
            accepted += 1; self.actor_steps += 1
        for _ in range(4):
            value_loss = (self.critic(x).squeeze(-1)-target).square().mean()
            self.critic_opt.zero_grad(); value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
            self.critic_opt.step(); self.critic_steps += 1
        assert all(torch.isfinite(p).all() for model in [self.actor, self.critic] for p in model.parameters())
        self.updates += 1
        return dict(actor_steps=accepted, rejected_kl_steps=rejected, critic_steps=4,
                    value_mse=float(value_loss.detach()), policy_loss=float(loss.detach()), measured_kl=float(kl),
                    candidate_tests=candidate_tests,accepted_step_scales=accepted_scales)

    def save(self, path):
        torch.save(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                        actor_opt=self.actor_opt.state_dict(), critic_opt=self.critic_opt.state_dict(),
                        mean=self.mean.cpu(), scale=self.scale.cpu(), updates=self.updates,
                        actor_steps=self.actor_steps, critic_steps=self.critic_steps,
                        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state()), path)

    def load(self, path):
        d = torch.load(path, map_location=self.device, weights_only=False)
        for key in ['actor', 'critic', 'actor_opt', 'critic_opt']:
            getattr(self, key).load_state_dict(d[key])
        self.mean = d['mean'].to(self.device); self.scale = d['scale'].to(self.device)
        self.updates = d['updates']; self.actor_steps = d['actor_steps']; self.critic_steps = d['critic_steps']
        return d
