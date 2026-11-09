"""Development-only sustained residual probes; frozen traction control unchanged."""
import argparse
import json
import sys
from pathlib import Path
import torch

p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
p.add_argument('--probe-joint',choices=['zero','hip_pitch','knee','ankle_pitch'],required=True)
p.add_argument('--probe-offset',type=float,default=0.)
option,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_traction_transfer_ppo as traction

class ProbeEnv(traction.TractionEnv):
    def step(self,actions):
        actions=torch.zeros_like(actions)
        if option.probe_joint!='zero':
            for i,n in enumerate(traction.progressive.base.BODY_JOINT_NAMES):
                if n in [s+'_'+option.probe_joint+'_joint' for s in ['left','right']]:
                    ratio=option.probe_offset/float(self.span[i]);assert abs(ratio)<1
                    actions[:,i]=torch.atanh(torch.tensor(ratio,device=self.device))
        return super().step(actions)
    def close(self):
        super().close()
        (self.out/'probe_contract.json').write_text(json.dumps(vars(option),indent=2))
        (self.out/'source'/'probe_posture_credit.py').write_text(Path(__file__).read_text())

traction.progressive.ProgressiveEnv=ProbeEnv
if __name__=='__main__':traction.progressive.main()
