"""Evaluate the three-second normal landing with frozen current policies.

No optimizer, new actor, or held-out geometry is used by this entry point.
"""
import argparse,json,sys,shutil
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
p.add_argument('--landing-config',required=True)
option,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_positive_slope_parent_reset_v2 as old
from low_seat_normal_timing_controller import install
assert old.early.early_mode=='evaluate'
adapter=SimpleNamespace(parent_harness=old.parent_harness,FeedbackEnv=old.PositiveEnv,old=old)
install(adapter,json.loads(Path(option.landing_config).read_text()))
old.PositiveEnv=adapter.FeedbackEnv
if __name__=='__main__':
    old.main()
    out=Path(sys.argv[sys.argv.index('--output')+1])
    for name in [Path(__file__).name,'low_seat_normal_timing_controller.py','low_seat_load_controller.py']:
        shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
