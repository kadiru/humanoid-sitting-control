"""Reference-only validation of geometry-limited rolling plus shear feedback."""
import argparse,sys,json
from pathlib import Path
p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--model-clearance',type=float,default=.006);p.add_argument('--model-gap',type=float,default=.001)
cfg,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_traction_transfer_probe as shear
from traction_clearance_planner_v2 import plan
class PlannedEnv(shear.TractionEnv):
    def close(self):
        super().close();(self.out/'planner_config.json').write_text(json.dumps(vars(cfg),indent=2))
        for name in ['traction_clearance_planner_v2.py','run_traction_planner_probe_v2.py']:(self.out/'source'/name).write_text(Path(__file__).with_name(name).read_text())
shear.progressive.rolling_plan=lambda *a,**kw:plan(*a,**kw,inertial=shear.bridge.inertial,clearance=cfg.model_clearance,gap=cfg.model_gap)
shear.progressive.ProgressiveEnv=PlannedEnv
if __name__=='__main__':shear.progressive.main()
