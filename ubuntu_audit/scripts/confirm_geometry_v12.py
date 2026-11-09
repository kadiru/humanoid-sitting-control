"""Additional untouched geometry bank after the representative videos finish."""
import subprocess,time,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
patterns=['^python3 scripts/run_geometry_v12_validation.py$','^python3 scripts/record_geometry_milestones.py$']
while any(subprocess.run(['pgrep','-f',p],stdout=subprocess.DEVNULL).returncode==0 for p in patterns):time.sleep(2)
parent='../baselines/rigid-furniture-medium-v1/policy_129_inputs.pt';pilot='../runs/geometry_reference_v11_ppo_pilot_01/model_79.pt'
for label,bank,seeds,flags in [('development_parent',9612,[2040,2041,2042],['--checkpoint',parent]),('scripted',10515,[3110,3111,3112],[]),('ppo',10515,[3110,3111,3112],['--checkpoint',parent]),('pilot',10515,[3110,3111,3112],['--checkpoint',pilot])]:
 name='geometry_reference_v12_'+('development_parent_01' if label=='development_parent' else 'confirmation_'+label+'_01')
 cmd=['bash','scripts/run.sh','python','../scripts/evaluate_geometry_reference.py','--v12-reference','--geometry-stage','2','--geometry-seed',str(bank),'--geometry-count','16','--families','--arm-clearance','--seeds',*map(str,seeds),'--output','../runs/'+name,'--device','cuda:0',*flags]
 with (root/'logs'/(name+'.log')).open('x') as f:code=subprocess.call(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
 if code:raise SystemExit(f'{name}: {code}')
 r=json.loads((root/'runs'/name/'results.json').read_text());print(label,r['success_rate'],r['maximum_seat_impact_N'],flush=True)
