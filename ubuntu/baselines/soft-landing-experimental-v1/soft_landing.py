"""Sensor-only contact inference and experimental recurrent phase governor.
No simulator imports. Privileged forces are labels outside this module.
"""
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
        m=cls().to(device);m.load_state_dict(torch.load(path,map_location=device,weights_only=False)['state_dict']);m.eval();return m

class SoftLandingController:
    """Explicit governor over frozen learned motion; this is not a fully learned policy.
    A knee encoder gate slows late descent; learned contact/load estimate controls
    transfer rate. Ankle neural feedback keeps running at full sensor frequency.
    """
    def __init__(self,policy,estimator,rate=.6,parameters=None,mode="phase",disable_contact_feedback=False):
        assert policy.variant=='geometry_recurrent'
        self.policy=policy;self.estimator=estimator;self.rate=rate
        self.contact_state=None;self.last_contact=None;self.last_rate=None;self.parameters=parameters;self.mode=mode;self.disable_contact_feedback=disable_contact_feedback
    def __call__(self,x,state):
        m=self.policy
        contact,self.contact_state=self.estimator(x,self.contact_state)
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
        if self.mode=="actions":rate=torch.ones_like(rate)
        state=old+(proposed-old)*rate.transpose(0,1)[...,None]
        latent=state.transpose(0,1)
        prediction=m.decoder(latent);prediction[...,15:17]=m.ankle_head(torch.cat((normalized,latent),-1))
        self.last_contact=measured_contact[:,0];self.last_rate=rate[:,0]
        targets=prediction*m.target_std+m.target_mean
        if self.mode=="actions":
            previous=x[...,75:98]
            ankle_rate=.004-.002*contact[...,0]
            for j in (15,16):
                guarded=previous[...,j]+(targets[...,j]-previous[...,j]).clamp(-ankle_rate,ankle_rate)
                targets[...,j]=torch.where(x[...,11:13].mean(-1)>1.0,guarded,targets[...,j])
            # Gradual transfer after inferred contact, without privileged load.
            for j in (0,1,11,12):
                guarded=previous[...,j]+(targets[...,j]-previous[...,j]).clamp(-.0025,.0025)
                targets[...,j]=torch.where(contact[...,0]>.65,guarded,targets[...,j])
        return targets,state
