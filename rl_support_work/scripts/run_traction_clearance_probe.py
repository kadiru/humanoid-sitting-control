"""Separate physical probes of a gentler contact transition and housing clearance.

Preserves the first ten seconds. All planner changes are explicit, common across
geometries, and logged; no geometry labels or simulator contacts enter control.
"""
import argparse,sys,json
from pathlib import Path
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--transfer-goal-x',type=float,default=-.16)
p.add_argument('--transfer-gap',type=float,default=.002)
p.add_argument('--settle-fraction',type=float,default=.2)
settings,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_traction_transfer_probe as shear
shear.bridge.static_args.goal_x=settings.transfer_goal_x
shear.bridge.static_args.contact_gap=settings.transfer_gap
shear.bridge.static_args.settle_fraction=settings.settle_fraction
class ClearanceTractionEnv(shear.TractionEnv):
    def close(self):
        super().close()
        (self.out/'clearance_traction_config.json').write_text(json.dumps(vars(settings),indent=2))
        (self.out/'source'/'run_traction_clearance_probe.py').write_text(Path(__file__).read_text())
shear.progressive.ProgressiveEnv=ClearanceTractionEnv
if __name__=='__main__':shear.progressive.main()
