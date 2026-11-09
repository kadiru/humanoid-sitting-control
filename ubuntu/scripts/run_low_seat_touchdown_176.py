"""One separately frozen zero-optimizer development probe of148."""
import argparse,json,sys,shutil
from pathlib import Path
p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
p.add_argument('--landing-config',required=True);p.add_argument('--fixed-config',required=True)
option,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_low_positive_feedback_v2 as harness
from low_seat_touchdown_runtime_176 import install
assert harness.options.feedback_mode=='zero'
install(harness,json.loads(Path(option.landing_config).read_text()),json.loads(Path(option.fixed_config).read_text()))
if __name__=='__main__':
    harness.main()
    out=Path(sys.argv[sys.argv.index('--output')+1])
    for name in [Path(__file__).name,'low_seat_touchdown_runtime_176.py','low_seat_balance_feedback_160.py','low_seat_touchdown_path_175.py']:
        shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    (out/'touchdown_wrapper_176.json').write_text(json.dumps(dict(
        change='176 retains164 measured balance/poses, uses175 joint-specific rate reservation and current-measured settling at12mm before6mm/s normal descent.',
        unchanged='150 geometry/loads,130 current guards,132 measured-view routing, original actuator filter/rate/material/physical criteria and frozen parent.',
        inherited_contract='fixed_contract_132.json describes the unchanged base; balance_contract_176.json describes the new bounded feedback and nominal rate allocation.',
        actual_sensors_only=True,optimizer_updates=0,physical_admission=False),indent=2))
