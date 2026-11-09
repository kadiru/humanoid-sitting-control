"""Isolated physical-controller experiment; no PPO update in this entry point."""
import argparse, json, sys
from pathlib import Path
p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
p.add_argument('--landing-config',required=True)
options,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_low_positive_feedback_v2 as harness
from low_seat_contact_controller import install
config=json.loads(Path(options.landing_config).read_text())
assert harness.options.feedback_mode in ['zero','reference'], 'Physical repair must precede training'
install(harness,config)
if __name__=='__main__':
    harness.main()
    import shutil
    out=Path(sys.argv[sys.argv.index('--output')+1])
    shutil.copy2(__file__,out/'source'/Path(__file__).name)
