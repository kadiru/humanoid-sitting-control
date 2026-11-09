"""Finite, persistent-action probes around the unchanged starting policy.

Exploration diagnosis only, not PPO or a claimed learned improvement.
The extra command is zero until the original 12-second approach finishes.
"""
import hashlib
import shutil
from pathlib import Path
import torch
import run_transfer_effort as original


class EffortProbe(original.SplitEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.offset=torch.zeros(self.num_envs,23,device=self.device)
        for i,j in enumerate(self.index):
            p=self.raw.cfg.geometry_bank[j]['control_profile']
            for side in ('left','right'):
                for name,v in [('knee',p.get('probe_knee',0.)),('hip_pitch',-.5*p.get('probe_knee',0.)),('ankle_pitch',-.5*p.get('probe_knee',0.)+p.get('probe_ankle',0.))]:
                    self.offset[i,original.base.BODY_JOINT_NAMES.index(side+'_'+name+'_joint')]=v
        source=Path(__file__)
        shutil.copy2(source,self.out/'effort_probe_source.py')
        original.base.write_json(self.out/'effort_probe_manifest.json',dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest(),offsets_rad=self.offset.cpu().tolist(),scope='Sensor/reference-only deterministic held offsets; original reference and physical criteria unchanged. No learned improvement claim.'))

    def step(self,actions):
        phase=((self.episode_length_buf*original.DT-12)/2).clamp(0,1)
        ramp=phase.square()*(3-2*phase)
        extra=self.offset*ramp[:,None]
        mixed=torch.atanh((torch.tanh(actions)+extra/self.span).clamp(-.999,.999))
        # Exactly reproduce the parent for zero-offset rows and during approach.
        use=(extra!=0).any(1)
        return super().step(torch.where(use[:,None],mixed,actions))


original.SplitEnv=EffortProbe
if __name__=='__main__':original.main()
