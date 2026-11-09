"""Physical development probe with unchanged contact rules and extra labels."""
import argparse,sys,json,shutil
from pathlib import Path
import numpy as np
extra=argparse.ArgumentParser(add_help=False);extra.add_argument('--positive-config',required=True)
option,rest=extra.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_posture_episode_ppo as harness
from positive_slope_planner import plan
config=json.loads(Path(option.positive_config).read_text())
harness.traction.progressive.rolling_plan=lambda *a,**kw:plan(*a,**kw,inertial=harness.traction.bridge.inertial,**config)

class DiagnosticEnv(harness.EpisodeEnv):
    def _physics_step(self,actions):
        result=super()._physics_step(actions)
        m,d,f=harness.support.metrics(self.raw);r=self.raw.scene['robot'];positions=[]
        for sensor,body,rest in harness.support.CONTACTS:
            positions.append(self.raw.scene[sensor].data.contact_pos_w.torch[:,0,0].cpu().numpy())
        self.contact_records[-1].update(legal=d['contacts_legal'].cpu().numpy(),inside=m['inside_patch'].cpu().numpy(),
            contact_position=np.stack(positions,1)-self.raw.scene.env_origins.cpu().numpy()[:,None,:],
            unexpected_body_normal=self.raw.scene['all_contacts'].data.net_normal_forces_w_history.torch.norm(dim=-1).max(1).values.cpu().numpy())
        return result
    def close(self):
        super().close()
        for name in ['run_positive_slope_reference.py','positive_slope_planner.py']:shutil.copy2(Path(__file__).with_name(name),self.out/'source'/name)
        (self.out/'positive_config.json').write_text(json.dumps(config,indent=2))
        (self.out/'contact_body_names.json').write_text(json.dumps(self.raw.scene['all_contacts'].body_names))
harness.EpisodeEnv=DiagnosticEnv
if __name__=='__main__':harness.main()
