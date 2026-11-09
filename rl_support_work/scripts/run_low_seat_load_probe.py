import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--landing-config',required=True)
options,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_low_positive_feedback_v2 as harness
from low_seat_load_controller import install
assert harness.options.feedback_mode in ['zero','reference']
install(harness,json.loads(Path(options.landing_config).read_text()))
if __name__=='__main__':
    harness.main()
    import shutil
    out=Path(sys.argv[sys.argv.index('--output')+1]);shutil.copy2(__file__,out/'source'/Path(__file__).name)
