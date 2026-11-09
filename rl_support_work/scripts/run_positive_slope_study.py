"""Small independent curricula, frozen budget, retained development stress set."""
import argparse,time,traceback
from positive_slope_common import *
from positive_slope_validation import gate
from run_transfer_assessment import run,status
STUDY=DATA/'ppo_01'

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True);a=p.parse_args();seed=SEEDS[a.gpu]
    protocol=read(STUDY/'protocol.json');assert protocol['admitted'];check(protocol['hashes']);check(protocol['prior_hashes'])
    folder=STUDY/f'seed_{seed}';folder.mkdir(exist_ok=False);state=STUDY/f'status_{seed}.json'
    try:
        start=9.;cp=None;records=[];candidates=[]
        for block in range(4):
            root=folder/f'block_{block:02d}';root.mkdir();status(state,'training',block=block,start_s=start,seed=seed)
            run(root,'train',command(root,a.gpu,'train','train',seed,start,cp,updates=3))
            cp=root/'train/final.pt';c=read(root/'train/early_contract.json');assert c['local_updates']==3 and c['active']==54000 and c['warmup']==900 and c['total']==54900 and c['final_updates']==3*(block+1)
            status(state,'current_stage_gate',block=block,start_s=start)
            run(root,'current_eval',command(root,a.gpu,'current_eval','evaluate',seed,start,cp))
            outcome=gate(root/'current_eval');record=dict(block=block,trained_start_s=start,checkpoint=str(cp),sha256=sha(cp),current_gate=outcome,current_evaluation=str(root/'current_eval'))
            def candidate(t,label,g):
                if g['passed']:candidates.append(dict(checkpoint=str(cp),sha256=sha(cp),start_s=t,evaluation=str(root/label),gate=g,block=block))
            candidate(start,'current_eval',outcome)
            if block<3 and outcome['passed'] and STAGES.index(start)<len(STAGES)-1:
                proposed=STAGES[STAGES.index(start)+1];status(state,'earlier_stage_gate',block=block,proposed_start_s=proposed)
                run(root,'earlier_eval',command(root,a.gpu,'earlier_eval','evaluate',seed,proposed,cp));next_gate=gate(root/'earlier_eval')
                record.update(proposed_start_s=proposed,earlier_gate=next_gate,earlier_evaluation=str(root/'earlier_eval'));candidate(proposed,'earlier_eval',next_gate)
                if next_gate['passed']:start=proposed
            records.append(record);write(folder/'curriculum.json',records)
        final=dict(checkpoint=str(cp),sha256=sha(cp),start_s=records[-1]['trained_start_s'],updates=12,timestamp=time.time(),training_evaluation=records[-1]['current_evaluation'])
        selected=min(candidates,key=lambda x:(x['start_s'],x['gate']['effort']['metrics_whole_trial']['policy'],x['gate']['effort']['metrics_last5s']['policy'])) if candidates else dict(parent_fallback=True,checkpoint=str(checkpoint(59117)),sha256=sha(checkpoint(59117)),start_s=11.)
        selected=dict(selected,selection_timestamp=time.time(),selection='Admitted development training surfaces only: earliest passing time, whole-motion/resting effort; explicit frozen-parent fallback. Known stress and holdout excluded.')
        write(folder/'selection.json',selected);write(STUDY/f'training_finished_{seed}.json',final)
        status(state,'waiting_for_other_final',**final)
        while not all((STUDY/f'training_finished_{s}.json').exists() for s in SEEDS):
            for s in SEEDS:
                p=STUDY/f'status_{s}.json';assert not p.exists() or read(p)['phase']!='failed'
            time.sleep(15)
        barrier=STUDY/'holdout_barrier.json'
        if a.gpu==0:write(barrier,dict(timestamp=time.time(),finals={str(s):read(STUDY/f'training_finished_{s}.json') for s in SEEDS},selections={str(s):read(STUDY/f'seed_{s}/selection.json') for s in SEEDS}))
        while not barrier.exists():time.sleep(5)
        jobs=[]
        if a.gpu==0:
            jobs.append(('development_parent_repaired',repaired_baseline(STUDY,0,'development_parent_repaired','59117','development')))
            for name in ['reference','59111']:
                label='holdout_'+name;jobs.append((label,baseline(STUDY,0,label,name,'holdout')))
            label='holdout_repaired_reference';jobs.append((label,repaired_baseline(STUDY,0,label,'reference','holdout')))
        else:
            for name in ['59117','60117']:
                label='holdout_'+name;jobs.append((label,baseline(STUDY,1,label,name,'holdout')))
            label='holdout_parent_repaired';jobs.append((label,repaired_baseline(STUDY,1,label,'59117','holdout')))
        distinct=not selected.get('parent_fallback') and (selected['checkpoint']!=final['checkpoint'] or selected['start_s']!=final['start_s'])
        for bank in ['development','holdout']:
            label=f'{bank}_final_{seed}';jobs.append((label,command(STUDY,a.gpu,label,'evaluate',seed,final['start_s'],cp,bank=bank)))
            if distinct:
                label=f'{bank}_selected_{seed}';jobs.append((label,command(STUDY,a.gpu,label,'evaluate',seed,selected['start_s'],selected['checkpoint'],bank=bank)))
        for label,cmd in jobs:status(state,'matched_final_comparison',active=label);run(STUDY,label,cmd)
        check(protocol['hashes']);check(protocol['prior_hashes']);status(state,'complete',**final,selected=selected,distinct_selected=distinct)
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
