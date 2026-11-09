"""Serialized, paired held-out comparison. Never overwrite existing runs."""
import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
checkpoint='../baselines/rigid-furniture-medium-v1/policy_129_inputs.pt'
conditions=[('development_scripted',['--geometry-seed','9612','--seeds','2040','2041','2042']),('scripted',[]),('ppo',['--checkpoint',checkpoint]),('ppo_blind',['--checkpoint',checkpoint,'--blind-geometry']),('canonical_ppo',['--checkpoint',checkpoint,'--reference-mode','canonical_geometry']),('legacy_ppo',['--checkpoint',checkpoint,'--legacy-reference','--reference-mode','gentle_recovery'])]
manifest=[]
for label,flags in conditions:
 name='geometry_reference_v11_holdout_'+label+'_01'
 cmd=['bash','scripts/run.sh','python','../scripts/evaluate_geometry_reference.py','--geometry-stage','2','--geometry-seed','10512','--geometry-count','16','--families','--arm-clearance','--seeds','3101','3102','3103','--output','../runs/'+name,'--device','cuda:0',*(['--v11-reference'] if label!='legacy_ppo' else []),*flags]
 with (root/'logs'/ (name+'.log')).open('x') as f:code=subprocess.call(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
 if code:raise SystemExit(f'{label} failed execution: {code}')
 r=json.loads((root/'runs'/name/'results.json').read_text());manifest.append(dict(label=label,run=name,command=cmd,success_rate=r['success_rate']))
 print(label,r['success_rate'],r['maximum_seat_impact_N'],flush=True)
 (root/'runs/geometry_reference_v11_comparison_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
