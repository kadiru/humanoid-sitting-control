"""Persistent leaning preflight. No full study starts without a passing gate."""
import json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
import torch
from run_transfer_assessment import run,write,status
from verify_split_leaning import load,equal,weights,starts

ROOT=Path('/home/kadir/humanoid-sitting')
DATA=Path('/data/humanoid-sitting-perception/contact_effort_repair_01')


def main():
    folder=DATA/'leaning_preflight_01';folder.mkdir(exist_ok=False)
    try:
        original=json.loads((DATA/'leaning_zero_child_01.command.json').read_text())
        original[2]='../scripts/run_stable_effort_leaning.py'
        def command(label,mode,checkpoint=None):
            c=original.copy();c[c.index('--output')+1]=str(folder/label);c[c.index('--mode')+1]=mode
            if mode=='train':c[c.index('--seed')+1]='56101';c+=['--iterations','16','--critic-warmup','8']
            if checkpoint:c+=['--checkpoint',str(checkpoint)]
            return c
        status(folder/'status.json','identity')
        run(folder,'zero',command('zero','evaluate'))
        parent=Path('/data/humanoid-sitting-perception/transfer_effort_01/assessment/leaning/policy_55311')
        for name in ('startup.npz','trace.npz'):
            a=np.load(folder/'zero'/name);b=np.load(parent/name)
            assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files),name
        status(folder/'status.json','optimizer_smoke')
        run(folder,'train',command('train','train'))
        initial=load(folder/'train/initial.pt');warm=load(folder/'train/critic_warmup.pt');final=load(folder/'train/final.pt')
        assert equal(initial['actor_state_dict'],warm['actor_state_dict'])
        changed=weights(initial,final)
        budget=json.loads((folder/'train/verification.json').read_text())
        assert budget['updates']==16 and budget['actor_updates']==8 and budget['policy_decisions']==4608 and budget['transitions']==46080
        assert budget['active']+budget['warmup']==46080
        status(folder/'status.json','smoke_evaluation')
        run(folder,'evaluate',command('evaluate','evaluate',folder/'train/final.pt'))
        actual=load(folder/'evaluate/initial.pt')
        assert all(equal(final[k],actual[k]) for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict','iter'))
        startup=starts(parent,folder/'evaluate')
        rows=json.loads((folder/'evaluate/results.json').read_text())['episodes']
        assert all(x['steps']==1000 and x['duration_s']==20 for x in rows)
        success=sum(x['success'] for x in rows);failures=sum(any(x['failures'].values()) for x in rows)
        check=dict(passed=success==18 and failures==0,successes=success,physical_failures=failures,exact_identity=True,startups=startup,training=budget,changed=changed,critic_warmup_actor_unchanged=True)
        write(folder/'verification.json',check)
        assert check['passed'],check
        status(folder/'status.json','preflight_passed')
    except Exception as e:
        status(folder/'status.json','failed',error=repr(e),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
