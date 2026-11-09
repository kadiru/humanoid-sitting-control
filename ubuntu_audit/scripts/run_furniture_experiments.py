"""Bounded stage probes with a strict promotion gate; all evidence preserved."""
import json
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[1]
checkpoint=root/'IsaacLab/logs/rsl_rl/g1_full_transition/2026-10-09_10-08-57_seed_7/model_96.pt'
def evaluate(name,*extra):
    command=[sys.executable,str(root/'scripts/evaluate_furniture.py'),'--device','cuda:0','--output',str(root/'runs'/name),*extra]
    with (root/'logs'/f'{name}.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    r=json.loads((root/'runs'/name/'results.json').read_text())
    print(name,json.dumps({k:r[k] for k in ('num_trials','success_rate','fall_rate','unintended_contact_rate','excessive_impact_rate','maximum_seat_impact_N','maximum_descent_m_s','minimum_seated_hold_s')}),flush=True)
    (root/'runs'/name/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    return r
r=evaluate('furniture_stage0_scripted_01','--geometry-stage','0')
passed=lambda r:r['success_rate']>=.9 and r['fall_rate']==0 and r['unintended_contact_rate']==0 and r['excessive_impact_rate']==0 and not any(o['excessive_descent'] or o['pelvis_floor'] for o in r['outcomes'])
if passed(r):
    evaluate('furniture_stage1_fixed_policy_01','--geometry-stage','1','--checkpoint',str(checkpoint))
    evaluate('furniture_stage1_scripted_01','--geometry-stage','1')
for mode in ['frozen_standing','phase_lag']:
    evaluate('furniture_stage0_ablation_'+mode+'_01','--geometry-stage','0','--reference-mode',mode,'--checkpoint',str(checkpoint))
evaluate('furniture_stage0_ablation_blind_geometry_01','--geometry-stage','0','--blind-geometry','--checkpoint',str(checkpoint))
evaluate('furniture_stage0_video_01','--geometry-stage','0','--checkpoint',str(checkpoint),'--video','--kit_args','--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false')
