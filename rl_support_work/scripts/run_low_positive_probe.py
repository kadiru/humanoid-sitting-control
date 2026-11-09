"""New physical probe; completed controllers and physical labels are unchanged."""
import argparse,sys,json,shutil
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--low-positive-config',required=True)
options,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_positive_slope_ppo_v4 as harness
from low_positive_planner import plan
config=json.loads(Path(options.low_positive_config).read_text())
harness.parent_harness.traction.progressive.rolling_plan=lambda *a,**kw:plan(*a,**kw,inertial=harness.parent_harness.traction.bridge.inertial,**config)
Parent=harness.PositiveEnv
class DiagnosticEnv(Parent):
    def _physics_step(self,actions):
        result=super()._physics_step(actions)
        m,d,f=harness.support.metrics(self.raw)
        self.contact_records[-1].update(contact_position=np.stack([self.raw.scene[s].data.contact_pos_w.torch[:,0,0].cpu().numpy() for s,b,r in harness.support.CONTACTS],1)-self.raw.scene.env_origins.cpu().numpy()[:,None,:],unexpected_body_normal=self.raw.scene['all_contacts'].data.net_normal_forces_w_history.torch.norm(dim=-1).max(1).values.cpu().numpy())
        return result
    def close(self):
        names=list(self.raw.scene['all_contacts'].body_names)
        super().close()
        (self.out/'contact_body_names.json').write_text(json.dumps(names))
        (self.out/'low_positive_config.json').write_text(json.dumps(config,indent=2))
        for name in ['run_low_positive_probe.py','low_positive_planner.py']:shutil.copy2(Path(__file__).with_name(name),self.out/'source'/name)
harness.PositiveEnv=DiagnosticEnv
if __name__=='__main__':harness.main()
