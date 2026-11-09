"""Fresh family holdout, reference comparison, videos, and gated wider probe."""
import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
cp=root/'IsaacLab/logs/rsl_rl/g1_full_transition/2026-10-09_10-08-57_seed_7/model_96.pt'
def evaluate(name,extra,stage=1,seed=9611):
 cmd=[sys.executable,str(root/'scripts/evaluate_furniture.py'),'--device','cuda:0','--geometry-stage',str(stage),'--reference-mode','gentle_recovery','--geometry-seed',str(seed),'--seeds','2040','2041','2042','--families','--arm-clearance','--geometry-count','16','--output',str(root/'runs'/name),*extra]
 with (root/'logs'/f'{name}.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 r=json.loads((root/'runs'/name/'results.json').read_text());print(name,{k:r[k] for k in ('num_trials','success_rate','fall_rate','unintended_contact_rate','excessive_impact_rate','maximum_seat_impact_N','maximum_descent_m_s','minimum_seated_hold_s')},flush=True)
 (root/'runs'/name/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');return r
passed=lambda r:r['success_rate']>=.9 and all(r[k]==0 for k in ['fall_rate','unintended_contact_rate','excessive_impact_rate','joint_torque_violation_rate']) and not any(o['excessive_descent'] or o['pelvis_floor'] for o in r['outcomes'])
a=evaluate('furniture_families_stage1_back_clearance_holdout_01',['--checkpoint',str(cp)])
b=evaluate('furniture_families_stage1_back_clearance_scripted_holdout_01',[])
if passed(a) and passed(b):
 for i in range(4):evaluate(f'furniture_family_{i}_video_04',['--video-geometry-index',str(i),'--checkpoint',str(cp),'--video','--kit_args','--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'])
 evaluate('furniture_families_stage2_probe_03',['--checkpoint',str(cp)],stage=2,seed=9612)
else:
 print('Family stage held: no geometry expansion.',flush=True)
