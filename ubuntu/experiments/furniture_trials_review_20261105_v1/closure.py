"""Read-only closure accounting for the twenty requested starts and preserved abort."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,subprocess,datetime
from pathlib import Path
ROOT=Path('/home/kadir/humanoid-sitting');OUT=ROOT/'outputs/furniture_trials_review_20261105_v1'
RUN=ROOT/'outputs/furniture_examples_20261105_v2';SRC=ROOT/'experiments/furniture_examples_20261105_v2'
CONT=ROOT/'outputs/furniture_examples_continuation_20261105_v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    status=read(CONT/'status.json');assert status['state']=='COMPLETE'
    result=read(OUT/'results.json');assert len(result['rows'])==20
    frozen=read(SRC/'freeze.json')['files']
    for p,h in frozen.items():assert sha(p)==h,('FROZEN_INPUT_CHANGED',p)
    active=warm=physics=0;rows=[]
    for row in result['rows']:
        folder=RUN/row['id'];exit=read(RUN/(row['id']+'.exit.json'))
        if row['task']=='sitting':
            capture=read(folder/'furniture_capture.json');startup=read(folder/'furniture_startup.json')
            a=capture['samples'];w=startup['warm_steps'];p=capture['physics_steps']
        else:
            d=read(folder/'results.json');a=d['active_transitions'];w=d['warmup_transitions'];p=d['actual_total_physics_steps']
        assert a==(1000 if row['full20s'] else 0) and w==25 and p==4*(a+w)
        active+=a;warm+=w;physics+=p
        rows.append(dict(id=row['id'],active_controls=a,warmup_controls=w,measured_physics_steps=p,exit=exit))
    ps=subprocess.check_output(['ps','-eo','pid,ppid,stat,args'],text=True)
    scientific=[line for line in ps.splitlines() if any(x in line for x in [str(SRC/'run_sitting.py'),str(SRC/'run_leaning.py'),str(ROOT/'experiments/furniture_examples_continuation_20261105_v1/resume.py')])]
    assert not scientific,('SCIENTIFIC_WORKER_STILL_LIVE',scientific)
    receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='ALL20_STARTS_PROCESSED',
        attempted_new_starts=20,complete20s_rollouts=result['full20s_rollouts'],planning_rejections_without_motion=result['planning_rejections_without_motion'],
        new_passes=result['new_passes'],failed_complete_rollouts=result['failed_complete_rollouts'],
        measured_active_controls=active,measured_warmup_controls=warm,measured_physics_steps=physics,
        preserved_reporting_abort=dict(active_controls=1,warmup_controls=25,inferred_physics_steps=104,source=str(ROOT/'outputs/furniture_examples_20261105_v1')),
        total_simulation_invocations_including_abort=21,total_active_controls_including_abort=active+1,total_warmup_controls_including_abort=warm+25,
        total_physics_steps_including_inferred_abort=physics+104,original_max_active_controls=20001,original_max_warmup_controls=525,
        frozen_source_input_hashes_verified=len(frozen),scientific_workers_live=False,gpu4070_used_by_trials=False,
        gpu4090_uuid='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4',ppo_updates=0,original72_retested=False,criteria_unchanged=True,
        owner_wall_s=read(ROOT/'outputs/furniture_examples_20261105_v1/status.json')['wall_s']+read(RUN/'status.json')['wall_s']+status['wall_s'],
        final_gpu_status=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,utilization.gpu,temperature.gpu,memory.used','--format=csv,noheader'],text=True),
        final_gpu_compute_workers=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],text=True),
        rows=rows,source_results_sha256=sha(OUT/'results.json'))
    assert receipt['owner_wall_s']<14000
    (OUT/'resource_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
