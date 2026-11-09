"""Finite-horizon PPO: one persistent posture choice, one measured full return.

No bootstrapped short-segment values or simulator labels enter the policy.
The value baseline and actor consume only the same frozen-normalized sensors.
"""
import copy
import torch
from torch import nn

class PosturePPO:
    def __init__(self,mean,scale,seed,device='cpu'):
        torch.manual_seed(seed)
        self.device=device
        self.mean=torch.as_tensor(mean,dtype=torch.float32,device=device)
        self.scale=torch.as_tensor(scale,dtype=torch.float32,device=device)
        self.actor=self.network(3).to(device);self.critic=self.network(1).to(device)
        for model in [self.actor,self.critic]:
            nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=.001)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=.001)
        self.std=.6;self.updates=0
    @staticmethod
    def network(n):return nn.Sequential(nn.Linear(414,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,n))
    def inputs(self,x):return ((x.to(self.device)-self.mean)/self.scale).clamp(-5,5)
    def distribution(self,x):
        mu=self.actor(self.inputs(x));return torch.distributions.Normal(mu,torch.full_like(mu,self.std))
    @torch.no_grad()
    def choose(self,x,stochastic=True):
        d=self.distribution(x);action=d.sample() if stochastic else d.mean
        return action,d.log_prob(action).sum(-1),self.critic(self.inputs(x)).flatten(),d.mean
    def update(self,obs,actions,old_logp,old_mean,reward):
        # Values are frozen before optimization. Whole-episode returns contain
        # every consequence of the one action, including terminal failures.
        with torch.no_grad():
            target=reward.detach();old_value=self.critic(self.inputs(obs)).flatten()
            raw_advantage=target-old_value
            advantage=(raw_advantage-raw_advantage.mean())/(raw_advantage.std(unbiased=False)+1e-8)
        steps=0;rejected=0;max_kl=0.;surrogate=0.
        for _ in range(8):
            old_actor=copy.deepcopy(self.actor.state_dict());old_opt=copy.deepcopy(self.actor_optimizer.state_dict())
            d=self.distribution(obs);ratio=(d.log_prob(actions).sum(-1)-old_logp).exp()
            loss=torch.maximum(-advantage*ratio,-advantage*ratio.clamp(.8,1.2)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(self.actor.parameters(),.5);self.actor_optimizer.step()
            with torch.no_grad():kl=((self.distribution(obs).mean-old_mean).square()/(2*self.std**2)).sum(-1).mean()
            if float(kl)>.02:
                self.actor.load_state_dict(old_actor);self.actor_optimizer.load_state_dict(old_opt);rejected+=1;break
            max_kl=max(max_kl,float(kl));steps+=1;surrogate=float(loss.detach())
        for _ in range(8):
            value=self.critic(self.inputs(obs)).flatten();loss=(value-target).square().mean()
            self.critic_optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(self.critic.parameters(),1.);self.critic_optimizer.step()
        self.updates+=1
        assert all(torch.isfinite(x).all() for m in [self.actor,self.critic] for x in m.parameters())
        return dict(actor_steps=steps,critic_steps=8,rejected_kl_steps=rejected,max_accepted_kl=max_kl,
                    surrogate=surrogate,value_mse=float(loss.detach()),reward_mean=float(target.mean()),
                    reward_std=float(target.std(unbiased=False)),raw_advantage_std=float(raw_advantage.std(unbiased=False)))
    def state(self):
        return dict(actor_state_dict=self.actor.state_dict(),critic_state_dict=self.critic.state_dict(),
                    actor_optimizer=self.actor_optimizer.state_dict(),critic_optimizer=self.critic_optimizer.state_dict(),
                    input_mean=self.mean,input_scale=self.scale,updates=self.updates,std=self.std,
                    contract='414 sensor/reference inputs; 3 persistent symmetric hip/knee/ankle offsets; full-episode PPO')
    def save(self,path):torch.save(self.state(),path)
    def load(self,path):
        x=torch.load(path,map_location=self.device,weights_only=False)
        self.mean=x['input_mean'];self.scale=x['input_scale'];self.std=x['std'];self.updates=x['updates']
        self.actor.load_state_dict(x['actor_state_dict']);self.critic.load_state_dict(x['critic_state_dict'])
        self.actor_optimizer.load_state_dict(x['actor_optimizer']);self.critic_optimizer.load_state_dict(x['critic_optimizer'])

def episode_return(rows,baseline):
    """Reward labels only. Constant baseline centers without changing optimum.

    Whole20s effort includes a locked11s prefix. Thus differences equal the
    change in learned11–20s effort. Units/scaling are fixed, never clipped.
    """
    values=[]
    for row,b in zip(rows,baseline):
        effort=row['metrics_whole_trial']['total_squared_torque_Nm2']
        cost=(effort-b)*20/(9*150)
        physical=any(row['failures'].values())
        # A physically failed or unsupported trajectory cannot win by going limp.
        penalty=20*physical+10*(not row['success'])+5*(row['slip_ticks']>0)+5*(row['illegal_ticks']>0)
        # Conservative margin labels remain reported. Prefix flags are fixed;
        # additional late violations are penalized by the caller's counters.
        penalty+=2*(row.get('post11_margin_ticks',0)>0)
        values.append(-cost-penalty)
    return values
