"""Run one bounded PPO stage after checking the previous measured promotion gate.

Invoke with scripts/run.sh so this inherits the pinned environment and GPU selection.
"""
import argparse,json,subprocess,sys
from datetime import datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',type=int,choices=range(4),required=True);p.add_argument('--checkpoint');p.add_argument('--previous-evaluation');p.add_argument('--iterations',type=int,default=20);p.add_argument('--num-envs',type=int,default=64)
a=p.parse_args()
if a.stage:
    if not a.previous_evaluation or not a.checkpoint:p.error('Later stages require --previous-evaluation and --checkpoint.')
    result=json.loads(Path(a.previous_evaluation).read_text())
    if result['stage']!=a.stage-1 or not result['promotion_passed']:
        p.error('Previous stage has not passed the measured promotion gate. Continue debugging/training that stage.')
    if not Path(a.checkpoint).is_file():p.error('Checkpoint does not exist.')
if not 1<=a.iterations<=100:p.error('Use 1–100 iterations for a bounded pilot.')
project=Path(__file__).resolve().parents[1]
out=project/'runs'/f'{datetime.now():%Y%m%d_%H%M%S}_curriculum_stage_{a.stage}'
out.mkdir(exist_ok=False)
command=['isaaclab','train','--rl_library','rsl_rl','--task',f'Humanoid-Sit-Curriculum-{a.stage}-G1','--num_envs',str(a.num_envs),'--max_iterations',str(a.iterations),'--device','cuda:0','--seed','42','--experiment_name','g1_seated_curriculum','--run_name',f'stage_{a.stage}_bounded']
if a.checkpoint:command+=['--checkpoint',str(Path(a.checkpoint).resolve())]
(out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
print(f'Executing stage {a.stage}; log: {out}/train.log',flush=True)
with (out/'train.log').open('w') as stream:
    completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
sys.exit(completed.returncode)
