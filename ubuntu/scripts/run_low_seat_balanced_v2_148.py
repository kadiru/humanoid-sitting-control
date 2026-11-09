"""One separately frozen zero-optimizer development probe of148."""
import argparse,json,sys,shutil
from pathlib import Path
p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
p.add_argument('--landing-config',required=True);p.add_argument('--fixed-config',required=True)
option,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_low_positive_feedback_v2 as harness
from low_seat_balanced_runtime_v2_148 import install
assert harness.options.feedback_mode=='zero'
install(harness,json.loads(Path(option.landing_config).read_text()),json.loads(Path(option.fixed_config).read_text()))
if __name__=='__main__':
    harness.main()
    out=Path(sys.argv[sys.argv.index('--output')+1])
    for name in [Path(__file__).name,'low_seat_balanced_runtime_v2_148.py','low_seat_balanced_force_v2_145.py']:
        shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    (out/'balanced_contract_148.json').write_text(json.dumps(dict(
        change='144 minimum squared corner loads with individual-foot5mm CoP reserve;145 current-input allocation installed by148.',
        unchanged='126 current-input poses,128 follower,130 guards,132 current-view/callback execution,original filter/rate/material/physical criteria and frozen parent.',
        inherited_contract='fixed_contract_132.json describes the unchanged base; actual load objective and sources are superseded by this supplement.',
        actual_sensors_only=True,optimizer_updates=0,physical_admission=False),indent=2))
