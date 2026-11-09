"""Continue preserved probes after fixing the actual-USD dimension audit."""
import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
cp=root/'IsaacLab/logs/rsl_rl/g1_full_transition/2026-10-09_10-08-57_seed_7/model_96.pt'
runs=[('furniture_stage1_scripted_02',['--geometry-stage','1'])]
for mode in ['frozen_standing','phase_lag']:
    runs.append(('furniture_stage0_ablation_'+mode+'_02',['--geometry-stage','0','--reference-mode',mode,'--checkpoint',str(cp)]))
runs.append(('furniture_stage0_ablation_blind_geometry_02',['--geometry-stage','0','--blind-geometry','--checkpoint',str(cp)]))
runs.append(('furniture_stage0_video_02',['--geometry-stage','0','--checkpoint',str(cp),'--video','--kit_args','--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false']))
for name,extra in runs:
    cmd=[sys.executable,str(root/'scripts/evaluate_furniture.py'),'--device','cuda:0','--output',str(root/'runs'/name),*extra]
    with (root/'logs'/f'{name}.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    r=json.loads((root/'runs'/name/'results.json').read_text());print(name,{k:r[k] for k in ('success_rate','fall_rate','unintended_contact_rate','maximum_seat_impact_N','maximum_descent_m_s','minimum_seated_hold_s')},flush=True)
    (root/'runs'/name/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
