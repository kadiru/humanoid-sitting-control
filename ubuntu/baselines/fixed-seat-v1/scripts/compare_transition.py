"""Execute fresh-seed validation and matched-camera before/after videos."""
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
full=root/'IsaacLab/logs/rsl_rl/g1_full_transition/2026-10-09_10-08-57_seed_7/model_96.pt'
old=root/'IsaacLab/logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-06-27_seated_stability/model_19.pt'
runs=[('seated_hold_policy_comparison_01',['--seated-baseline','--checkpoint',str(old)]),
      ('full_transition_policy_holdout_01',['--checkpoint',str(full),'--seeds','2024','2025','2026']),
      ('full_transition_policy_video_01',['--checkpoint',str(full),'--video']),
      ('seated_hold_policy_video_comparison_01',['--seated-baseline','--checkpoint',str(old),'--video'])]
commands=[]
for name,extra in runs:
    out=root/'runs'/name
    command=[sys.executable,str(root/'scripts/evaluate_full_transition.py'),'--device','cuda:0','--output',str(out),*extra]
    if '--video' in extra:command+=['--kit_args','--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false']
    commands.append(command)
    with (root/'logs'/f'{name}.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((out/'results.json').read_text());print('COMPARISON_COMPLETED',name,json.dumps({k:result[k] for k in ('success_rate','fall_rate','unintended_contact_rate','maximum_seat_impact_N','minimum_seated_hold_s')}),flush=True)
    subprocess.run([sys.executable,str(root/'scripts/plot_scripted.py'),str(out)],check=True)
(root/'runs/full_transition_comparison_commands_01.json').write_text(json.dumps(commands,indent=2)+'\n')
