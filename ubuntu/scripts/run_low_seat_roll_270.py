"""Bounded raw roll-velocity269 feedback on the unchanged complete centered233 path."""
import argparse,json,sys,shutil
from pathlib import Path
p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
p.add_argument('--landing-config',required=True);p.add_argument('--fixed-config',required=True)
option,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_low_positive_feedback_v2 as harness
from low_seat_roll_runtime_270 import install
assert harness.options.feedback_mode=='zero'
install(harness,json.loads(Path(option.landing_config).read_text()),json.loads(Path(option.fixed_config).read_text()))
if __name__=='__main__':
    harness.main()
    out=Path(sys.argv[sys.argv.index('--output')+1])
    for name in [Path(__file__).name,'low_seat_roll_runtime_270.py','low_seat_roll_feedback_269.py','low_seat_bias_feedback_v2_207.py','low_seat_touchdown_path_175.py']:
        shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    (out/'roll_wrapper_270.json').write_text(json.dumps(dict(
        change='270 retains233 complete centered227 path and adds269 bounded raw ankle-roll velocity feedback. Other rows use guardednormal3s. Unchanged207 proportional/integral pitch feedback and175 current settling guard; no saved inputs enter runtime.',
        unchanged='126 path construction,145 smooth force objective,130 guards,207 bounded antiwindup,175 raw-velocity settling, original filter/rate/material/physical criteria and frozen parent. Inherited contracts describing150 poses are historical; this contract and fixed_config identify the new centered geometry.',
        inherited_contract='fixed_contract_132.json describes the unchanged base; bias_contract_208.json describes the new bounded feedback and nominal rate allocation.',
        actual_sensors_only=True,optimizer_updates=0,physical_admission=False),indent=2))
