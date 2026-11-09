"""Sensor-only contact inference and experimental recurrent phase governor.
No simulator imports. Privileged forces are labels outside this module.
"""
from collections import deque
import torch
from torch import nn

class ContactEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean',torch.zeros(109));self.register_buffer('std',torch.ones(109))
        self.encoder=nn.Sequential(nn.Linear(109,64),nn.SiLU())
        self.rnn=nn.GRU(64,64,batch_first=True);self.head=nn.Linear(64,2)
    def forward(self,x,state=None):
        z=((x-self.mean)/self.std).clamp(-10,10).clone();z[...,106:108]=0
        z,state=self.rnn(self.encoder(z),state)
        out=self.head(z)
        return torch.stack((out[...,0].sigmoid(),out[...,1].sigmoid()),-1),state
    @classmethod
    def load(cls,path,device):
        m=cls().to(device);data=torch.load(path,map_location=device,weights_only=False);m.load_state_dict(data['state_dict']);m.window=data.get('window');m.eval();return m

class SoftLandingController:
    """Explicit governor over frozen learned motion; this is not a fully learned policy.
    A knee encoder gate slows late descent; learned contact/load estimate controls
    transfer rate. Ankle neural feedback keeps running at full sensor frequency.
    """
    def __init__(self,policy,estimator,rate=.6,parameters=None,mode="phase",disable_contact_feedback=False):
        assert policy.variant=='geometry_recurrent'
        self.policy=policy;self.estimator=estimator;self.rate=rate
        self.contact_window=deque(maxlen=getattr(estimator,"window",None) or 1)
        self.contact_state=None;self.last_contact=None;self.last_rate=None;self.parameters=parameters;self.mode=mode;self.disable_contact_feedback=disable_contact_feedback
    def __call__(self,x,state):
        m=self.policy
        if getattr(self.estimator,"window",None):
            self.contact_window.append(x)
            sequence=torch.cat(list(self.contact_window),dim=1)
            if sequence.shape[1]<self.estimator.window:
                sequence=torch.cat((sequence[:,:1].expand(-1,self.estimator.window-sequence.shape[1],-1),sequence),dim=1)
            contact,_=self.estimator(sequence,None);contact=contact[:,-1:]
        else:contact,self.contact_state=self.estimator(x,self.contact_state)
        measured_contact=contact.clone()
        if self.disable_contact_feedback:contact=torch.zeros_like(contact)
        normalized=((x-m.mean)/m.std).clamp(-20,20).clone()
        if m.mask_perception_status:normalized[...,106:108]=0
        _,proposed=m.rnn(m.encoder(normalized[...,98:]),state)
        old=torch.zeros_like(proposed) if state is None else state
        late=(x[...,11:13].mean(-1)>1.2)
        # Lower progression near the seat; positive estimated contact slows
        # initial transfer further, then permits gradual seated recovery.
        transfer=.45+.3*contact[...,1]
        if self.parameters is not None:
            base=self.parameters[:,0,None];transfer=self.parameters[:,1,None]*(.8+.2*contact[...,1])
        else:base=torch.full_like(transfer,self.rate)
        rate=torch.where(late,torch.where(contact[...,0]>.65,transfer,base),torch.ones_like(transfer))
        if self.mode in ("actions","smooth"):rate=torch.ones_like(rate)
        state=old+(proposed-old)*rate.transpose(0,1)[...,None]
        latent=state.transpose(0,1)
        prediction=m.decoder(latent);prediction[...,15:17]=m.ankle_head(torch.cat((normalized,latent),-1))
        self.last_contact=measured_contact[:,0];self.last_rate=rate[:,0]
        targets=prediction*m.target_std+m.target_mean
        if self.mode in ("actions","smooth"):
            previous=x[...,75:98]
            ankle_rate=.004-.002*contact[...,0]
            for j in (15,16):
                guarded=previous[...,j]+(targets[...,j]-previous[...,j]).clamp(-ankle_rate,ankle_rate)
                targets[...,j]=torch.where(x[...,11:13].mean(-1)>1.0,guarded,targets[...,j])
            # Gradual transfer after inferred contact, without privileged load.
            for j in (0,1,11,12):
                guarded=previous[...,j]+(targets[...,j]-previous[...,j]).clamp(-.0025,.0025)
                targets[...,j]=torch.where(contact[...,0]>.65,guarded,targets[...,j])
            if self.mode=="smooth":
                # Additional continuous encoder gate; original guard retained.
                # Lower ankle slew throughout final descent; no simulated forces.
                gate=((x[...,11:13].mean(-1)-.85)/.25).clamp(0,1)
                for j in (15,16):
                    delta=targets[...,j]-previous[...,j]
                    cap=(1-gate)*.05+gate*.0015
                    targets[...,j]=previous[...,j]+delta.clamp(-cap,cap)
        return targets,state

class StagedAnkleHead(nn.Module):
    """Experimental physical-posture gate; preserved learned standing controller.
    Both heads consume the same sensor packet and recurrent latent. No teacher,
    clock or simulator contact is used. This gate is explicitly authored.
    """
    def __init__(self,learned,baseline,mean,std):
        super().__init__();self.learned=learned;self.baseline=baseline
        self.register_buffer('knee_mean',mean[11:13].detach().clone());self.register_buffer('knee_std',std[11:13].detach().clone())
    def forward(self,x):
        knee=(x[...,11:13]*self.knee_std+self.knee_mean).mean(-1,keepdim=True)
        blend=((knee-1.0)/.2).clamp(0,1)
        return (1-blend)*self.baseline(x)+blend*self.learned(x)
