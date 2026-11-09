"""Separate physical timing probe; imports completed base sources unchanged."""
import argparse,sys,shutil,json
from pathlib import Path
p=argparse.ArgumentParser(add_help=False);p.add_argument('--transfer-duration',type=float,default=20./3)
options,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
out=Path(rest[rest.index('--output')+1])
import run_positive_slope_reference_v2 as parent
import positive_timed_reference as timing
timing.DURATION=options.transfer_duration
bridge=parent.harness.traction.bridge
bridge.BridgeEnv._positive_original_split_reference=lambda self:bridge.progressive.original.SplitEnv.reference(self)
bridge.BridgeEnv.reference=timing.timed_reference
if __name__=='__main__':
    parent.harness.main()
    for name in ['run_positive_slope_timed_reference.py','positive_timed_reference.py']:shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    (out/'timing_contract.json').write_text(json.dumps(dict(positive_transfer_duration_s=timing.DURATION,mild_transfer_duration_s=5.,first10s_preserved=True,source='Sensor/kinematic positive-candidate classification; no truth control'),indent=2))
