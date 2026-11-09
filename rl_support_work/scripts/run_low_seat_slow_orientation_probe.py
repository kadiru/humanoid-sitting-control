import argparse,json,sys,shutil
from pathlib import Path
p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--landing-config',required=True)
option,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_low_positive_feedback_v2 as harness
from low_seat_slow_orientation_controller import install
assert harness.options.feedback_mode=='zero'
install(harness,json.loads(Path(option.landing_config).read_text()))
if __name__=='__main__':
    harness.main()
    out=Path(sys.argv[sys.argv.index('--output')+1])
    for name in [Path(__file__).name,'low_seat_slow_orientation_controller.py','low_seat_normal_timing_controller.py']:
        shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
