"""Bounded warm-start pilot and paired evaluations, serialized on GPU 0."""
import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
parent='../baselines/rigid-furniture-medium-v1/policy_129_inputs.pt'
def run(name,script,flags):
 cmd=['bash','scripts/run.sh','python','../scripts/'+script,'--output','../runs/'+name,'--device','cuda:0',*flags]
 with (root/'logs'/(name+'.log')).open('x') as f:code=subprocess.call(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
 if code:raise SystemExit(f'{name}: execution failure {code}')
 print('COMPLETE',name,flush=True)
 if (root/'runs'/name/'results.json').exists():
  r=json.loads((root/'runs'/name/'results.json').read_text());print(r['success_rate'],r['maximum_seat_impact_N'],flush=True)
def evaluate(name,bank,seeds,flags):
 run(name,'evaluate_geometry_reference.py',['--v11-reference','--geometry-stage','2','--geometry-count','16','--geometry-seed',str(bank),'--families','--arm-clearance','--seeds',*map(str,seeds),*flags])
evaluate('geometry_reference_v11_development_ppo_01',9612,[2040,2041,2042],['--checkpoint',parent])
run('geometry_reference_v11_ppo_pilot_01','train_geometry_reference.py',['--checkpoint',parent,'--geometry-stage','2','--geometry-seed','41012','--seed','71','--iterations','80','--families','--arm-clearance'])
checkpoint='../runs/geometry_reference_v11_ppo_pilot_01/model_79.pt'
evaluate('geometry_reference_v11_holdout_pilot_01',10512,[3101,3102,3103],['--checkpoint',checkpoint])
evaluate('geometry_reference_v11_development_pilot_01',9612,[2040,2041,2042],['--checkpoint',checkpoint])
for label,flags in [('scripted',[]),('ppo',['--checkpoint',parent]),('pilot',['--checkpoint',checkpoint]),('pilot_blind',['--checkpoint',checkpoint,'--blind-geometry']),('canonical_pilot',['--checkpoint',checkpoint,'--reference-mode','canonical_geometry'])]:
 evaluate('geometry_reference_v11_final_'+label+'_01',10513,[3104,3105,3106],flags)
