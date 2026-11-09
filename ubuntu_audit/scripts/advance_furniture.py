"""Validate the slower recovery on untouched seeds before adding rigid families."""
import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
cp=root/'IsaacLab/logs/rsl_rl/g1_full_transition/2026-10-09_10-08-57_seed_7/model_96.pt'
def evaluate(name,extra):
 cmd=[sys.executable,str(root/'scripts/evaluate_furniture.py'),'--device','cuda:0','--geometry-stage','1','--reference-mode','gentle_recovery','--geometry-seed','9311','--seeds','2024','2025','2026','--output',str(root/'runs'/name),*extra]
 with (root/'logs'/f'{name}.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 r=json.loads((root/'runs'/name/'results.json').read_text());print(name,{k:r[k] for k in ('num_trials','success_rate','fall_rate','unintended_contact_rate','excessive_impact_rate','maximum_seat_impact_N','maximum_descent_m_s','minimum_seated_hold_s')},flush=True)
 (root/'runs'/name/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');return r
passed=lambda r:r['success_rate']>=.9 and all(r[k]==0 for k in ['fall_rate','unintended_contact_rate','excessive_impact_rate','joint_torque_violation_rate']) and not any(o['excessive_descent'] or o['pelvis_floor'] for o in r['outcomes'])
a=evaluate('furniture_stage1_gentle_scripted_holdout_01',[])
b=evaluate('furniture_stage1_gentle_fixed_policy_holdout_01',['--checkpoint',str(cp)])
if passed(a) and passed(b):
 c=evaluate('furniture_families_stage1_fixed_policy_01',['--families','--geometry-count','16','--checkpoint',str(cp)])
 if passed(c):
  for i in range(4):evaluate(f'furniture_family_{i}_video_01',['--families','--geometry-count','16','--video-geometry-index',str(i),'--checkpoint',str(cp),'--video','--kit_args','--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'])
