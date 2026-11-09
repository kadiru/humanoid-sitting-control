"""Bounded learned recurrent progression; sensor-only and simulator independent."""
import math
import torch
from .soft_landing import SoftLandingController


def update_pace(previous, action, ready):
    """IID PPO actions become smooth actuator requests, with observable filter state.

    Zero is exactly the old progression. Positive requests saturate at normal
    speed; negative requests may slow to 25%. The first-order filter has a
    0.19 s time constant at 50 Hz, with an additional 0.04/tick slew bound.
    No random process is hidden from PPO's log-probability calculation.
    """
    desired=(1.+.75*action.clamp(-1.,1.)).clamp(.25,1.)
    updated=previous+(.1*(desired-previous)).clamp(-.04,.04)
    return torch.where(ready,updated.clamp(.25,1.),torch.ones_like(updated))


class PaceController(SoftLandingController):
    """Actions-mode scaffold with a per-robot rate applied before decoding.

    This controls recurrent progression, not a guaranteed physical descent speed.
    Contact inference and ankle feedback still execute at the full sensor rate.
    """
    progress_rate=None

    def __call__(self,x,state):
        m=self.policy
        if getattr(self.estimator,'window',None):
            self.contact_window.append(x)
            sequence=torch.cat(list(self.contact_window),dim=1)
            if sequence.shape[1]<self.estimator.window:
                sequence=torch.cat((sequence[:,:1].expand(-1,self.estimator.window-sequence.shape[1],-1),sequence),dim=1)
            contact,_=self.estimator(sequence,None);contact=contact[:,-1:]
        else:contact,self.contact_state=self.estimator(x,self.contact_state)
        normalized=((x-m.mean)/m.std).clamp(-20,20).clone()
        if m.mask_perception_status:normalized[...,106:108]=0
        _,proposed=m.rnn(m.encoder(normalized[...,98:]),state)
        old=torch.zeros_like(proposed) if state is None else state
        rate=torch.ones_like(x[...,0]) if self.progress_rate is None else self.progress_rate[:,None]
        assert torch.isfinite(rate).all() and (rate>=.25).all() and (rate<=1.).all()
        state=old+(proposed-old)*rate.transpose(0,1)[...,None]
        latent=state.transpose(0,1)
        prediction=m.decoder(latent);prediction[...,15:17]=m.ankle_head(torch.cat((normalized,latent),-1))
        self.last_contact=contact[:,0].clone();self.last_rate=rate[:,0].clone()
        targets=prediction*m.target_std+m.target_mean
        previous=x[...,75:98]
        ankle_rate=.004-.002*contact[...,0]
        for j in (15,16):
            guarded=previous[...,j]+(targets[...,j]-previous[...,j]).clamp(-ankle_rate,ankle_rate)
            targets[...,j]=torch.where(x[...,11:13].mean(-1)>1.,guarded,targets[...,j])
        for j in (0,1,11,12):
            guarded=previous[...,j]+(targets[...,j]-previous[...,j]).clamp(-.0025,.0025)
            targets[...,j]=torch.where(contact[...,0]>.65,guarded,targets[...,j])
        return targets,state


def migrate_state(old,template,actor):
    """144/23 -> 145/24; original outputs preserved, new input disconnected.

    The new normalized input is progression rate (initial mean 1, std 1).
    The new output mean is zero; only its exploration std starts at 0.5.
    Optimizer history is intentionally not resized or silently reused.
    """
    result={k:v.clone() for k,v in template.items()}
    assert old.keys()==result.keys()
    for key,value in old.items():
        dst=result[key]
        if value.shape==dst.shape:dst.copy_(value);continue
        if key.startswith('obs_normalizer.'):
            assert value.shape==(1,144) and dst.shape==(1,145)
            dst.fill_(1.);dst[:,:144]=value
        elif key=='mlp.0.weight':
            assert value.shape==(128,144) and dst.shape==(128,145)
            dst.zero_();dst[:,:144]=value
        elif actor and key=='mlp.4.weight':
            assert value.shape==(23,128) and dst.shape==(24,128)
            dst.zero_();dst[:23]=value
        elif actor and key=='mlp.4.bias':
            assert value.shape==(23,) and dst.shape==(24,)
            dst.zero_();dst[:23]=value
        elif actor and key=='distribution.log_std_param':
            assert value.shape==(23,) and dst.shape==(24,)
            dst.fill_(math.log(.5));dst[:23]=value
        else:raise ValueError((key,value.shape,dst.shape))
    return result
