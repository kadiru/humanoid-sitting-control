import argparse,json,sys,shutil
from pathlib import Path
p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--landing-config',required=True)
option,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_low_positive_feedback_v2 as harness
from low_seat_joint_path_controller_17 import install
assert harness.options.feedback_mode=='zero'
install(harness,json.loads(Path(option.landing_config).read_text()))
if __name__=='__main__':
    harness.main()
    out=Path(sys.argv[sys.argv.index('--output')+1]);shutil.copy2(__file__,out/'source'/Path(__file__).name)
