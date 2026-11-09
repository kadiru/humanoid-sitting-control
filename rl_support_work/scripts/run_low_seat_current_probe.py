"""Apply an isolated landing repair after the current policy's early pulse.

The current early actor and resting actor are frozen. This entry point performs
one full physical evaluation and no optimizer update.
"""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
p.add_argument('--landing-config',required=True)
option,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_positive_slope_ppo_v4 as old
from low_seat_contact_controller import install
assert old.early.early_mode=='evaluate'
adapter=SimpleNamespace(parent_harness=old.parent_harness,FeedbackEnv=old.PositiveEnv,old=old)
install(adapter,json.loads(Path(option.landing_config).read_text()))
old.PositiveEnv=adapter.FeedbackEnv
if __name__=='__main__':
    old.main()
    import shutil
    out=Path(sys.argv[sys.argv.index('--output')+1])
    shutil.copy2(__file__,out/'source'/Path(__file__).name)
