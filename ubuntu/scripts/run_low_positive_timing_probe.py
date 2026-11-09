"""Separate timing probe, with all failures and full contact traces retained."""
import sys
import run_low_positive_probe as probe
from low_positive_timing import plan
probe.harness.parent_harness.traction.progressive.rolling_plan=lambda *a,**kw:plan(*a,**kw,inertial=probe.harness.parent_harness.traction.bridge.inertial,**probe.config)
if __name__=='__main__':
    probe.harness.main()
    from pathlib import Path
    import shutil
    out=Path(sys.argv[sys.argv.index('--output')+1])
    for name in ['low_positive_timing.py','run_low_positive_timing_probe.py']:shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
