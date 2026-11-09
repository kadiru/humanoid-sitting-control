"""Independent fixed-budget, source, optimizer, motion and outcome audit."""
import argparse,shutil
import numpy as np
from positive_slope_common_v2 import *
from positive_slope_validation_v2 import physical,matched,actors,replay,gate
from verify_seat_load_study import evaluation,compare,load,equal
from verify_posture_credit_study import contacts

PREFLIGHT=DATA/'preflight_v2'
def paths(study,bank):
    if bank=='development':
        p={'reference':DATA/'assessment/reference','parent_59111':DATA/'assessment/59111','parent_59117':DATA/'assessment/59117','current_60117':DATA/'assessment/60117','repaired_reference':DATA/'repair_06/reference','repaired_parent':study/'development_parent_repaired'}
    else:
        p={k:study/('holdout_'+v) for k,v in [('reference','reference'),('parent_59111','59111'),('parent_59117','59117'),('current_60117','60117'),('repaired_reference','repaired_reference'),('repaired_parent','parent_repaired')]}
    for seed in SEEDS:
        p[f'final_{seed}']=study/f'{bank}_final_{seed}'
        final=read(study/f'training_finished_{seed}.json');chosen=read(study/f'seed_{seed}/selection.json')
        if not chosen.get('parent_fallback') and (chosen['checkpoint'],chosen['start_s'])!=(final['checkpoint'],final['start_s']):p[f'selected_{seed}']=study/f'{bank}_selected_{seed}'
    return p

def sources(folder,protocol):
    m=read(folder/'manifest.json')
    for p,h in m['sources'].items():assert protocol['hashes'][p]==h,(folder,p)
    for name,h in m['bank_hashes'].items():assert sha(Path(m['args']['bank'])/name)==h
    for p in (folder/'source').iterdir():
        key=str(ROOT/'scripts'/p.name)
        if key in protocol['hashes']:assert sha(p)==protocol['hashes'][key],p
    if 'selected_reference_sha256' in m:assert m['selected_reference_sha256']==sha(DATA/'selected_reference_v2.json')

def clean(rows):return [dict(x,success=bool(x['success'] and not any(x['failures'].values()) and not x['slip_ticks'] and not x['illegal_ticks'])) for x in rows]
def summarize(rows):
    return dict(trials=len(rows),stable=sum(x['success'] for x in rows),clean=sum(x['success'] for x in clean(rows)),physical=sum(any(x['failures'].values()) for x in rows),slipping=sum(x['slip_ticks']>0 for x in rows),unwanted_contact=sum(x['illegal_ticks']>0 for x in rows),strict=sum(x['strict_transfer'] for x in rows),failure_labels={k:sum(x['failures'][k] for x in rows) for k in rows[0]['failures']} if rows else {})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=False)
    protocol=read(a.study/'protocol.json');assert sha(a.study/'protocol.json')==(a.study/'protocol.sha256').read_text().strip();check(protocol['hashes']);check(protocol['prior_hashes'])
    barrier=read(a.study/'holdout_barrier.json');out=dict(passed=False,source_hashes=len(protocol['hashes']),prior_hashes=len(protocol['prior_hashes']),training={},selections={},paths={},evaluations={},comparisons={},strict_comparisons={},actors={},startups={},contacts={})
    out['preflight_accounting']={}
    for rootname in ['preflight','preflight_v2','reset_recovery_01']:
        total=dict(active=0,warmup=0,initial_settle=0,actor_steps=0,critic_steps=0);items={}
        for folder in sorted((DATA/rootname).iterdir()):
            if not folder.is_dir():continue
            contract=folder/'early_contract.json' if (folder/'early_contract.json').exists() else folder/'episode_contract.json'
            if not contract.exists():continue
            c=read(contract);audit=read(folder/'learning_audit.json')
            item=dict(active=c['active'],warmup=c['warmup'],initial_settle=c['initial_settle_excluded'],actor_steps=sum(x['optimizer'].get('actor_steps',0) for x in audit),critic_steps=sum(x['optimizer'].get('critic_steps',0) for x in audit))
            assert item['active']==1000*len(read(folder/'results.json')['episodes'])
            items[folder.name]=item
            for k in total:total[k]+=item[k]
        out['preflight_accounting'][rootname]=dict(totals=total,runs=items,separate_from_main=True)
    for seed in SEEDS:
        folder=a.study/f'seed_{seed}';records=read(folder/'curriculum.json');assert len(records)==4;start=9.;previous=None;candidates=[];totals=dict(active=0,warmup=0,initial_settle=0,actor_steps=0,critic_steps=0,new_actor_applied_ticks=0,frozen_parent_applied_ticks=0);stage_updates={}
        for i,row in enumerate(records):
            assert row['block']==i and row['trained_start_s']==start
            train=folder/f'block_{i:02d}/train';c=read(train/'early_contract.json')
            assert c['active']==54000 and c['warmup']==900 and c['total']==54900 and c['episodes']==54 and c['initial_settle_excluded']==450 and c['local_updates']==3 and c['final_updates']==3*(i+1)
            sources(train,protocol);actors(train,previous);rr=replay(train,3)
            decision=np.load(train/'episode_decisions.npz');plans=read(train/'rolling_plan_audit.json');assert len(plans)==54
            planned=np.array([150 if x.get('positive_candidate_applied') else 110 for x in plans]).reshape(3,18)
            assert np.array_equal(decision['handover_step'],planned)
            early_ticks=int(np.rint(((planned/10-start)/.02).sum()));parent_ticks=int(np.rint(((20-planned/10)/.02).sum()))
            assert early_ticks+parent_ticks==round((20-start)/.02)*54
            totals['new_actor_applied_ticks']+=early_ticks;totals['frozen_parent_applied_ticks']+=parent_ticks
            acc=read(train/'control_accounting.json');assert acc['excluded_ticks']==0 and acc['active_physics']==54000 and acc['indexed_warmup']==900
            assert acc['control_ticks']==round((20-start)/.02)*54 and acc['teacher_ticks']==round(start/.02)*54
            assert all(abs(t-start)<1e-6 for t in acc['earliest_control_s'])
            assert all(x['steps']==1000 for x in read(train/'results.json')['episodes'])
            if i==0:assert equal(load(train/'initial.pt'),load(PREFLIGHT/f'smoke_{seed}/initial.pt'))
            for key in ['active','warmup']:totals[key]+=c[key]
            totals['initial_settle']+=450
            for key in ['actor_steps','critic_steps']:totals[key]+=rr[key]
            stage_updates[str(start)]=stage_updates.get(str(start),0)+3
            cp=train/'final.pt';assert str(cp)==row['checkpoint'] and sha(cp)==row['sha256']
            current=gate(Path(row['current_evaluation']));assert current==row['current_gate']
            tests=[('current_evaluation',start,current)]
            if 'proposed_start_s' in row:
                assert i<3 and current['passed'] and row['proposed_start_s']==STAGES[STAGES.index(start)+1]
                g=gate(Path(row['earlier_evaluation']));assert g==row['earlier_gate'];tests.append(('earlier_evaluation',row['proposed_start_s'],g))
            for label,t,g in tests:
                p=Path(row[label]);sources(p,protocol);actors(p,cp);evaluation(p,18);physical(p,require_clean=False);matched(PREFLIGHT/'reference',p,t)
                if g['passed']:candidates.append(dict(checkpoint=str(cp),sha256=sha(cp),start_s=t,evaluation=str(p),gate=g,block=i))
            if len(tests)>1 and tests[-1][2]['passed']:start=tests[-1][1]
            previous=cp
        assert totals['active']==216000 and totals['warmup']==3600 and totals['initial_settle']==1800
        final=read(a.study/f'training_finished_{seed}.json');chosen=read(folder/'selection.json')
        assert final==barrier['finals'][str(seed)] and final['sha256']==sha(previous) and final['timestamp']<barrier['timestamp']
        assert chosen==barrier['selections'][str(seed)] and chosen['selection_timestamp']<barrier['timestamp']
        if candidates:
            expected=min(candidates,key=lambda x:(x['start_s'],x['gate']['effort']['metrics_whole_trial']['policy'],x['gate']['effort']['metrics_last5s']['policy']))
            for k,v in expected.items():assert chosen[k]==v
        else:assert chosen['parent_fallback'] and chosen['sha256']==sha(checkpoint(59117))
        out['training'][str(seed)]=dict(**totals,physics_transitions=totals['active']+totals['warmup'],stage_updates=stage_updates,final_start_s=final['start_s'],CPU_optimizer_replay=True,zero_unfinished_active_ticks=True,zero_consumed_pending_warmup=True,initial_equals_smoke=True)
        out['selections'][str(seed)]=chosen
    for bank in ['development','holdout']:
        pp=paths(a.study,bank);out['paths'][bank]={k:str(p) for k,p in pp.items()};allrows={};geoms={}
        for name,p in pp.items():
            summary,rows,g=evaluation(p,24);key=bank+'_'+name
            outcome=physical(p,require_clean=False);out['evaluations'][key]=dict(**outcome,checks=summary)
            out['contacts'][key]=contacts(p);out['startups'][key]=matched(pp['reference'],p,6.)
            allrows[name]=rows;geoms[name]=g
            if bank=='development':
                admitted=[x for x in rows if x['geometry_index'] in [0,1,4,5,6,8,9,10,11]]
                # Evaluation geometry_index can index per-placement expanded geometry.
                admitted=[x for x in rows if not ((g[x['geometry_index']]['height']<.38 and g[x['geometry_index']]['pitch_deg']>=15) or (abs(g[x['geometry_index']]['height']-.4)<1e-6 and g[x['geometry_index']]['pitch_deg']>=20))]
                stress=[x for x in rows if x not in admitted];assert len(admitted)==18 and len(stress)==6
                out['evaluations'][key].update(admitted=summarize(admitted),stress=summarize(stress))
            if name.startswith(('final_','selected_')):
                seed=int(name.split('_')[-1]);which=read(a.study/f'training_finished_{seed}.json') if name.startswith('final') else read(a.study/f'seed_{seed}/selection.json')
                sources(p,protocol);out['actors'][key]=actors(p,Path(which['checkpoint']));out['startups'][key]=matched(pp['reference'],p,which['start_s'])
            if bank=='holdout':assert (p/'manifest.json').stat().st_mtime>barrier['timestamp']
        baselines=['reference','parent_59111','parent_59117','current_60117','repaired_reference','repaired_parent']
        for name in allrows:
            if name=='reference':continue
            out['comparisons'][bank+'_'+name]={b:compare(clean(allrows[name]),clean(allrows[b]),geoms[name]) for b in baselines}
            out['strict_comparisons'][bank+'_'+name]={b:compare([x for x in clean(allrows[name]) if x['strict_transfer']],clean(allrows[b]),geoms[name]) for b in baselines}
    out['scope']='Whole20/final10/final5 torque comparisons require common full-duration supported successes with zero all-tick slip and unwanted contact in BOTH controllers; strict additionally requires policy conservative-margin clean. Torque is an effort proxy, not energy. Nearby rigid slabs and two seeds are limited evidence.'
    check(protocol['hashes']);check(protocol['prior_hashes']);out['passed']=True;write(a.output/'verification.json',out);shutil.copy2(__file__,a.output/'verification_source.py')
    print(dict(passed=True,training=out['training'],outcomes={k:(v['stable'],v['clean'],v['physical']) for k,v in out['evaluations'].items()}))
if __name__=='__main__':main()
