"""Matched selected-reference or immutable-parent evaluation, with diagnostics."""
import argparse,sys,json,shutil
from pathlib import Path
extra=argparse.ArgumentParser(add_help=False);extra.add_argument('--positive-reference',required=True)
option,rest=extra.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_posture_episode_ppo as harness
from positive_slope_runtime import install
selected=install(harness,option.positive_reference)
if __name__=='__main__':
    harness.main()
    out=Path(sys.argv[sys.argv.index('--output')+1])
    for name in ['run_positive_selected_baseline.py','positive_slope_runtime.py','positive_slope_planner.py','positive_timed_reference.py']:shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    (out/'selected_reference.json').write_text(json.dumps(selected,indent=2))
