"""Evaluation-only recovery for new holdout labels absent from reward calibration.

Never trains or changes a frozen source. All original outputs remain in place.
The missing scale is used only to compute an unused evaluation reward; it is
not an observation, action, success criterion or reported torque metric.
"""
import json,time,shutil,traceback
from pathlib import Path
import numpy as np
import run_stable_effort_comparison as study
from run_transfer_assessment import run,write,status

folder=study.DATA/'leaning_ppo_01'
while True:
    state=study.read(folder/'status.json')
    if state['phase']=='verified_complete':raise SystemExit('No recovery needed')
    if state['phase']=='failed':break
    time.sleep(30)
failed_log=folder/'holdout_final_56111.log'
assert failed_log.exists(),state
text=failed_log.read_text()
assert 'KeyError' in text and 'lean_dh' in text and not (folder/'holdout_final_56111/results.json').exists(),state
assert (folder/'selection_frozen.json').exists()
recovery=folder/'execution_recovery_01';recovery.mkdir(exist_ok=False)
write(recovery/'original_failure_status.json',state)
shutil.copy2(__file__,recovery/'recovery_source.py')
original=study.read(study.PREVIOUS/'calibration/leaning.json')
cal=dict(original);pooled=float(np.mean(list(original.values())))
geometry=study.read(study.DATA/'banks/holdout/geometry.json')
added=[]
for g in geometry:
    if g['kind']=='lean' and g['label'] not in cal:
        cal[g['label']]=pooled;added.append(g['label'])
write(recovery/'evaluation_calibration.json',cal)
write(recovery/'manifest.json',dict(reason='Frozen training calibration contains development labels only; evaluator attempted a lookup for a new holdout label.',repair='For evaluation only, fill missing labels with the mean of the frozen development scales. No holdout outcome is used. Runtime actor/controller, metrics, success gates and training are unchanged.',added_labels=added,scale=pooled,original_calibration=study.sha(study.PREVIOUS/'calibration/leaning.json'),original_training_complete=True,source=study.sha(__file__)))
out=recovery/'evaluations';out.mkdir();view=recovery/'view';view.mkdir()
jobs=[('holdout_reference','reference',None,None,True)]
for seed,oldseed in zip(study.SEEDS,study.PARENTS):
    s=study.read(folder/f'seed_{seed}.json');par=study.parent(oldseed)
    jobs.extend([(f'holdout_parent_{oldseed}','evaluate',None,par,True),(f'holdout_final_{seed}','evaluate',par,s['final_checkpoint'],False)])
    if s['selected_checkpoint'] and s['selected_checkpoint']!=s['final_checkpoint']:
        jobs.append((f'holdout_selected_{seed}','evaluate',par,s['selected_checkpoint'],False))
mapping={}
try:
    for label,mode,par,cp,legacy in jobs:
        existing=folder/label
        if (existing/'results.json').exists():mapping[label]=str(existing);continue
        cmd=study.command(out,label,mode,55501,par,cp,bank='holdout',legacy=legacy)
        if not legacy:cmd[cmd.index('--calibration')+1]=str(recovery/'evaluation_calibration.json')
        status(recovery/'status.json','evaluating',active=label)
        run(recovery,label,cmd);mapping[label]=str(out/label)
    for p in folder.iterdir():
        if p.name=='execution_recovery_01' or p.name.startswith('verified_'):continue
        (view/p.name).symlink_to(Path(mapping.get(p.name,str(p))))
    for label,target in mapping.items():
        if not (view/label).exists():(view/label).symlink_to(target)
    write(recovery/'output_mapping.json',mapping)
    status(recovery/'status.json','verifying')
    study.verify(view,recovery/'verified_01')
    status(recovery/'status.json','verified_complete')
    status(folder/'status.json','verified_complete_execution_recovery',recovery=str(recovery),view=str(view))
except Exception as e:
    status(recovery/'status.json','failed',error=repr(e),traceback=traceback.format_exc());raise
