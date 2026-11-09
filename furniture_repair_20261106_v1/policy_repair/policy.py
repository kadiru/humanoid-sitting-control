"""23 residuals; protected routes use the exact prior during training and evaluation.

The mask applies to the entire action distribution. A protected route is a
deterministic action, with zero log-density contribution and no PPO actor loss.
The critic still learns from its transitions. Original action limits are intact.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from common import OBS,STEP,FILTER,MARGIN

def command(previous,latent,lower,upper):
    desired=latent.clamp(lower+MARGIN,upper-MARGIN)
    return previous+(FILTER*(desired-previous)).clamp(-STEP,STEP)

def teacher_targets(previous,applied,lower,upper):
    return (previous+(applied-previous)/FILTER).clamp(lower+MARGIN,upper-MARGIN)

def advantages(reward,value,next_value,done,gamma=.999,lam=.97):
    result=torch.zeros_like(reward);carry=torch.zeros_like(reward[0])
    for t in range(len(reward)-1,-1,-1):
        continuation=(~done[t]).float()
        carry=reward[t]+gamma*next_value[t]*continuation-value[t]+gamma*lam*continuation*carry
        result[t]=carry
    return result,result+value

class Learner:
    def __init__(self,mean,scale,seed,device='cpu'):
        torch.manual_seed(seed);self.device=device
        self.mean=torch.as_tensor(mean,dtype=torch.float32,device=device)
        self.scale=torch.as_tensor(scale,dtype=torch.float32,device=device).clamp_min(.05)
        assert self.mean.shape==self.scale.shape==(OBS,)
        def net(n):return nn.Sequential(nn.Linear(OBS,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,n)).to(device)
        self.actor=net(23);self.critic=net(1)
        self.actor[-1].weight.data.zero_();self.actor[-1].bias.data.zero_()
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=1e-4)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=3e-4)
        self.std=.0075;self.updates=self.actor_steps=self.critic_steps=0
        self.library=None;self.sitting=None;self.demo=None

    def normalize(self,obs):return ((obs-self.mean)/self.scale).clamp(-10,10)

    def bind_library(self,library):
        self.library={k:torch.as_tensor(v,device=self.device) for k,v in library.items()}

    def bind_sitting(self,library):
        self.sitting={k:torch.as_tensor(v,device=self.device) for k,v in library.items()}

    def prior(self,obs,details=False):
        assert self.library is not None
        signature=obs[:,336:]
        dist=(((signature[:,None]-self.library['signature'][None])/self.library['signature_scale'])**2).mean(-1)
        choice=dist.argmin(-1)
        tick=(obs[:,335]*1000).round().long().clamp(0,999)
        desired=self.library['desired'][choice,tick]
        sitting=obs[:,333]>.5
        if sitting.any():
            assert self.sitting is not None
            d=(((signature[:,None]-self.sitting['signature'][None])/self.sitting['signature_scale'])**2).mean(-1)
            sel=d.argmin(-1)
            desired=torch.where(sitting[:,None],self.sitting['desired'][sel,tick],desired)
        return (desired,choice,dist) if details else desired

    def residual_mask(self,obs):
        _,choice,_=self.prior(obs,True)
        return (obs[:,334:335]>.5).float()*self.library['new_skill'][choice,None]

    @torch.no_grad()
    def choose(self,obs,stochastic=True):
        x=self.normalize(obs);mask=self.residual_mask(obs)
        assert bool(((mask==0)|(mask==1)).all())
        mu=self.prior(obs)+self.actor(x)*mask;dist=Normal(mu,self.std)
        a=torch.where(mask.bool(),dist.sample(),mu) if stochastic else mu
        logp=(dist.log_prob(a)*mask).sum(-1)
        return a,logp,self.critic(x).squeeze(-1),mu

    def bind_demonstrations(self,obs,targets=None):
        self.demo=self.normalize(torch.as_tensor(obs,dtype=torch.float32,device=self.device))

    def update(self,obs,action,logp,old_mean,advantage,target):
        x=self.normalize(obs);prior=self.prior(obs).detach();mask=self.residual_mask(obs).detach()
        assert bool(((mask==0)|(mask==1)).all())
        active=mask[:,0].bool()
        adv=torch.zeros_like(advantage)
        if active.any():
            values=advantage[active]
            adv[active]=(values-values.mean())/values.std(unbiased=False).clamp_min(1e-6)
        ix=torch.randint(len(self.demo),(2048,),device=self.device);dx=self.demo[ix]
        records=[];accepted=0;reasons={'kl':0,'objective':0,'nonfinite':0}
        def objective():
            mu=prior+self.actor(x)*mask
            ratio=((Normal(mu,self.std).log_prob(action)*mask).sum(-1)-logp).exp()
            ppo=-torch.minimum(ratio[active]*adv[active],ratio[active].clamp(.8,1.2)*adv[active]).mean() if active.any() else mu.sum()*0.
            imitation=(self.actor(dx)/.03).square().mean()
            return ppo+1.0*imitation,ppo,imitation,mu
        for epoch in range(4 if active.any() else 0):
            before=copy.deepcopy(self.actor.state_dict());opt=copy.deepcopy(self.actor_opt.state_dict())
            loss,ppo,imitation,mu=objective();initial=float(loss.detach())
            self.actor_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.actor.parameters(),.5);self.actor_opt.step()
            proposed=copy.deepcopy(self.actor.state_dict());admitted=False
            with torch.no_grad():
                for exponent in range(8):
                    alpha=2.**(-exponent)
                    self.actor.load_state_dict({k:before[k]+alpha*(proposed[k]-before[k]) for k in before})
                    candidate,cp,ci,cm=objective();kl=((cm[active]-old_mean[active])**2/(2*self.std**2)).sum(-1).mean()
                    checks=dict(nonfinite=not bool(torch.isfinite(candidate)&torch.isfinite(kl)),kl=bool(kl>.02),objective=bool(candidate>initial))
                    row=dict(epoch=epoch,scale=alpha,objective=float(candidate),initial_objective=initial,kl=float(kl),imitation=float(ci),rejected_by=[k for k,v in checks.items() if v])
                    records.append(row)
                    if not row['rejected_by']:admitted=True;break
                    for k,v in checks.items():reasons[k]+=int(v)
            if not admitted:
                self.actor.load_state_dict(before);self.actor_opt.load_state_dict(opt);break
            accepted+=1;self.actor_steps+=1
        for _ in range(4):
            value_loss=(self.critic(x).squeeze(-1)-target).square().mean()
            self.critic_opt.zero_grad();value_loss.backward();nn.utils.clip_grad_norm_(self.critic.parameters(),1.);self.critic_opt.step();self.critic_steps+=1
        self.updates+=1
        with torch.no_grad():
            retained,p,i,m=objective();kl=((m[active]-old_mean[active])**2/(2*self.std**2)).sum(-1).mean() if active.any() else torch.zeros((),device=self.device)
        assert all(torch.isfinite(p).all() for model in [self.actor,self.critic] for p in model.parameters())
        return dict(actor_steps=accepted,critic_steps=4,policy_loss=float(retained),ppo_surrogate_loss=float(p),imitation_loss=float(i),imitation_coefficient=1.0,
            measured_kl=float(kl),value_mse=float(value_loss.detach()),candidate_tests=records,rejection_counts=reasons,hard_imitation_ceiling=False,
            active_actor_samples=int(active.sum()),protected_samples=int((~active).sum()))

    def save(self,path):
        torch.save(dict(action_representation='preserved_route_distribution_v2',actor=self.actor.state_dict(),critic=self.critic.state_dict(),
            actor_opt=self.actor_opt.state_dict(),critic_opt=self.critic_opt.state_dict(),mean=self.mean.cpu(),scale=self.scale.cpu(),
            library={k:v.cpu() for k,v in self.library.items()},sitting={k:v.cpu() for k,v in self.sitting.items()},updates=self.updates,actor_steps=self.actor_steps,critic_steps=self.critic_steps),path)

    def load(self,path):
        d=torch.load(path,map_location=self.device,weights_only=False)
        assert d['action_representation'] in ('preserved_sitting_new_rest_residual_v1','preserved_route_distribution_v2')
        # Old optimized checkpoints are kept as evidence, never resumed with a
        # different action distribution. An untouched prior can be imported.
        if d['action_representation']=='preserved_sitting_new_rest_residual_v1':
            assert d['updates']==d['actor_steps']==d['critic_steps']==0,'OLD_TRAINING_CANNOT_RESUME'
        for k in ['actor','critic','actor_opt','critic_opt']:getattr(self,k).load_state_dict(d[k])
        self.mean=d['mean'].to(self.device);self.scale=d['scale'].to(self.device);self.bind_library(d['library']);self.bind_sitting(d['sitting'])
        self.updates=d['updates'];self.actor_steps=d['actor_steps'];self.critic_steps=d['critic_steps'];return d
