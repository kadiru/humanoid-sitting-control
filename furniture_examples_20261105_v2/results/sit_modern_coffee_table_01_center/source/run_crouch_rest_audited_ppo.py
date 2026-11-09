"""Record the actual post-adaptation actor input before any physics step."""
from pathlib import Path
import numpy as np
import run_crouch_rest_safe_ppo as bridge

class AuditedBridge(bridge.BridgeEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        obs=self.get_observations()
        np.savez_compressed(self.out/'bridge_startup.npz',policy=obs['policy'].cpu().numpy().copy(),learn=obs['learn'].cpu().numpy().copy(),reference=self.reference().cpu().numpy().copy())
    def close(self):
        super().close();(self.out/'source'/'run_crouch_rest_audited_ppo.py').write_text(Path(__file__).read_text())

bridge.progressive.ProgressiveEnv=AuditedBridge
if __name__=='__main__':bridge.progressive.main()
