"""Reference probe with every-tick contact/command audit, unchanged controller."""
from pathlib import Path
import json
import numpy as np
import run_crouch_rest_static_feedback as feedback
probe=feedback.probe

class AuditedEnv(feedback.FeedbackEnv):
    def __init__(self,*args,**kwargs):
        self.contact_ticks=[];super().__init__(*args,**kwargs)
    def _physics_step(self,actions):
        result=super()._physics_step(actions)
        m,d,f=probe.progressive.support.metrics(self.raw)
        self.contact_ticks.append(dict(t=float(self.episode_length_buf[0])*.02,force=m['force_w'].cpu().numpy(),slip=m['slip_m_s'].cpu().numpy(),normal=m['normal_w'].cpu().numpy(),weight=m['bodyweight_N'].cpu().numpy(),inside=m['inside_patch'].cpu().numpy(),wrong=d['wrong_force_N'].cpu().numpy(),command=self.previous.cpu().numpy().copy(),q=self.robot.data.joint_pos.torch.cpu().numpy().copy(),failure=m['failure'].cpu().numpy(),no_slip=d['no_slip'].cpu().numpy(),traction=d['traction_ok'].cpu().numpy()))
        return result
    def close(self):
        super().close();np.savez_compressed(self.out/'every_tick.npz',**{k:np.array([r[k] for r in self.contact_ticks]) for k in self.contact_ticks[0]});(self.out/'source'/'run_crouch_rest_audited.py').write_text(Path(__file__).read_text())

probe.progressive.ProgressiveEnv=AuditedEnv
if __name__=='__main__':probe.progressive.main()
