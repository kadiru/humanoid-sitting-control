"""Serialize representative videos after validation; leave all earlier movies intact."""
import subprocess,time,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
while subprocess.run(['pgrep','-f','^python3 scripts/run_geometry_v12_validation.py$'],stdout=subprocess.DEVNULL).returncode==0:time.sleep(2)
parent='../baselines/rigid-furniture-medium-v1/policy_129_inputs.pt'
for label,bank,index,policy in [('low_chair_parent',9612,4,parent),('shallow_box_scripted',9612,2,None),('heldout_couch_scripted',10514,3,None)]:
 name='geometry_reference_v12_'+label+'_video_01'
 cmd=['bash','scripts/run.sh','python','../scripts/evaluate_geometry_reference.py','--v12-reference','--geometry-stage','2','--geometry-seed',str(bank),'--geometry-count','16','--families','--arm-clearance','--seeds','3107','--video','--video-geometry-index',str(index),'--output','../runs/'+name,'--device','cuda:0','--kit_args=--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false']
 if policy:cmd.extend(['--checkpoint',policy])
 with (root/'logs'/(name+'.log')).open('x') as f:code=subprocess.call(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
 if code:raise SystemExit(f'{name}: {code}')
 r=json.loads((root/'runs'/name/'results.json').read_text());print(name,r['success_rate'],r['maximum_seat_impact_N'],flush=True)
 cmd=['bash','scripts/run.sh','python','../scripts/plot_scripted.py','../runs/'+name]
 with (root/'logs'/('plot-'+name+'.log')).open('x') as f:code=subprocess.call(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
 if code:raise SystemExit('Plot failed')
