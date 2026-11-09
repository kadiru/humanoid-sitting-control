"""Evaluate and verify the three bounded full-transition seed runs, preserving evidence."""
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
results=[]
for seed in (7,42,123):
    matches=sorted((root/'IsaacLab/logs/rsl_rl/g1_full_transition').glob(f'*_seed_{seed}'))
    run=matches[-1]
    checkpoint=max(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    output=root/'runs'/f'full_transition_policy_seed_{seed}_eval_01'
    with (root/'logs'/f'full-transition-policy-seed-{seed}-eval-01.log').open('x') as log:
        subprocess.run([sys.executable,str(root/'scripts/evaluate_full_transition.py'),
            '--checkpoint',str(checkpoint),'--seeds','7','42','123',
            '--output',str(output),'--device','cuda:0'],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,str(root/'scripts/verify_curriculum_checkpoint.py'),str(run),
        '--reference',str(root/'IsaacLab/logs/rsl_rl/g1_seated_curriculum/2026-10-09_00-20-27_partial_descent/model_57.pt')],check=True)
    result=json.loads((output/'results.json').read_text())
    results.append(dict(training_seed=seed,evaluation=str(output),**{k:result[k] for k in
        ('checkpoint','num_trials','success_rate','fall_rate','unintended_contact_rate',
         'maximum_seat_impact_N','minimum_seated_hold_s','median_seated_hold_s','maximum_seated_hold_s')}))
    print('SEED_EVALUATED',json.dumps(results[-1]),flush=True)
(root/'runs/full_transition_policy_batch_01.json').write_text(json.dumps(results,indent=2)+'\n')
