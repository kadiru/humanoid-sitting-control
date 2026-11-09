"""Persistent optimizer-smoke admission; does not launch the main training study."""
import json,time,traceback,hashlib
from pathlib import Path
import numpy as np
import torch
from run_crouch_rest_safe_study import DATA,ROOT,command,gate,read,sha
from run_transfer_assessment import run,write,status

def main():
    state=DATA/'preflight_status.json'
    try:
        status(state,'waiting_for_reference_and_reset_audits')
        files=[DATA/'safe_env_startup_reference_01/results.json',DATA/'safe_env_reset_memory_02/reset_probe.json']
        while not all(p.exists() for p in files):
            for label in ['safe_env_startup_reference_01','safe_env_reset_memory_02']:
                log=DATA/(label+'.log')
                assert not log.exists() or 'Traceback (most recent call last)' not in log.read_text(),label
            time.sleep(10)
        assert read(files[1])['passed'];assert read(DATA/'safe_env_regression_verification_01.json')['passed']
        ref=DATA/'safe_env_startup_reference_01';assert gate(ref)['passed']
        old=np.load(DATA/'foot_balance3_01/trace.npz');new=np.load(ref/'trace.npz')
        assert all(np.array_equal(old[k],new[k]) for k in old.files)
        write(DATA/'preflight_contract_01.json',dict(smoke_scheduled_updates=6,physics_transitions=17280,learning_start_s=15,gate='All12 admitted strict successes; no new physical-failure condition on other6; no holdout used',main_weights_fresh=True,source_hashes={str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in ['run_crouch_rest_preflight.py','run_crouch_rest_audited_ppo.py','run_crouch_rest_safe_ppo.py','crouch_rest_validated_controller.py','sagittal_static_bias.py']}))
        status(state,'optimizer_smoke',active='safe_ppo_smoke_01')
        run(DATA,'safe_ppo_smoke_01',command(DATA,0,'safe_ppo_smoke_01','train',15,58101,updates=6,warmup=0))
        train=DATA/'safe_ppo_smoke_01';cp=train/'final.pt';initial=torch.load(train/'initial.pt',map_location='cpu',weights_only=False);final=torch.load(cp,map_location='cpu',weights_only=False)
        roles={}
        for role in ['actor','critic']:
            x=initial[role+'_state_dict'];y=final[role+'_state_dict'];assert all(torch.isfinite(v).all() for v in y.values());assert any(not torch.equal(x[k],v) for k,v in y.items());roles[role]=True
        v=read(train/'verification.json');assert v['transitions']==17280
        c=read(train/'control_accounting.json');assert c['control_ticks']>0 and c['active_physics']==c['control_ticks']+c['teacher_ticks']+c['excluded_ticks']
        assert all(c['control_by_case'][i]==0 for i in [6,7,12,13,14,15])
        status(state,'optimizer_smoke_evaluation',active='safe_ppo_smoke_evaluation_01')
        run(DATA,'safe_ppo_smoke_evaluation_01',command(DATA,0,'safe_ppo_smoke_evaluation_01','evaluate',15,checkpoint=cp))
        evaluation=DATA/'safe_ppo_smoke_evaluation_01';g=gate(evaluation);assert g['passed'],g
        base_rows=read(ref/'results.json')['episodes'];rows=read(evaluation/'results.json')['episodes'];assert not any(any(x['failures'].values()) and not any(y['failures'].values()) for x,y in zip(rows,base_rows))
        for name in ['startup.npz','bridge_startup.npz']:
            a=np.load(ref/name);b=np.load(evaluation/name);assert all(np.array_equal(a[k],b[k]) for k in a.files),name
        a=np.load(ref/'trace.npz');b=np.load(evaluation/'trace.npz');mask=a['t']<=15
        assert all(np.array_equal(a[k][mask],b[k][mask]) for k in ['q','body','command','reference','torque'])
        write(DATA/'safe_ppo_admission_01.json',dict(passed=True,gate=g,finite_changed=roles,smoke_transitions=17280,smoke_checkpoint=sha(cp),exact_startup_and_teacher_prefix=True,control_accounting=c,main_training_started=False))
        status(state,'passed',admission=str(DATA/'safe_ppo_admission_01.json'))
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
