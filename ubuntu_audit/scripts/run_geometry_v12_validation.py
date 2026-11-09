"""Freeze a common safer lean rule; test before any promotion or further training."""
import subprocess,time,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
while subprocess.run(['pgrep','-f','^python3 scripts/run_geometry_v11_pilot.py$'],stdout=subprocess.DEVNULL).returncode==0:time.sleep(2)
parent='../baselines/rigid-furniture-medium-v1/policy_129_inputs.pt';pilot='../runs/geometry_reference_v11_ppo_pilot_01/model_79.pt'
cases=[('development_scripted',9612,[2040,2041,2042],[]),('development_pilot',9612,[2040,2041,2042],['--checkpoint',pilot]),('scripted',10514,[3107,3108,3109],[]),('ppo',10514,[3107,3108,3109],['--checkpoint',parent]),('pilot',10514,[3107,3108,3109],['--checkpoint',pilot]),('pilot_blind',10514,[3107,3108,3109],['--checkpoint',pilot,'--blind-geometry']),('canonical_pilot',10514,[3107,3108,3109],['--checkpoint',pilot,'--reference-mode','canonical_geometry'])]
manifest=[]
for label,bank,seeds,flags in cases:
 name='geometry_reference_v12_'+('development_' if label.startswith('development_') else 'holdout_')+label.removeprefix('development_')+'_01'
 cmd=['bash','scripts/run.sh','python','../scripts/evaluate_geometry_reference.py','--v12-reference','--geometry-stage','2','--geometry-count','16','--geometry-seed',str(bank),'--families','--arm-clearance','--seeds',*map(str,seeds),'--output','../runs/'+name,'--device','cuda:0',*flags]
 with (root/'logs'/(name+'.log')).open('x') as f:code=subprocess.call(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
 if code:raise SystemExit(f'{name}: {code}')
 r=json.loads((root/'runs'/name/'results.json').read_text());manifest.append(dict(label=label,run=name,command=cmd,success_rate=r['success_rate'],peak_force_N=r['maximum_seat_impact_N']))
 (root/'runs/geometry_reference_v12_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(label,r['success_rate'],r['maximum_seat_impact_N'],flush=True)
