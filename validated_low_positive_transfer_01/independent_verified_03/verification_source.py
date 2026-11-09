"""Independent frozen-study, startup, selection and matched-effort verification."""
import argparse,shutil
import numpy as np
from low_positive_common import *
from positive_slope_common_v5 import sha,write,check
from validate_low_positive_feedback_v3 import validate
from run_low_positive_study import gate,clean
from positive_slope_validation_v4 import physical,matched,actors
from verify_seat_load_study import evaluation,compare,load,equal

def summary(rows):
    return dict(trials=len(rows),stable=sum(r['success'] for r in rows),clean=sum(clean(r) for r in rows),physical=sum(any(r['failures'].values()) for r in rows),slipping=sum(r['slip_ticks']>0 for r in rows),unwanted_contact=sum(r['illegal_ticks']>0 or r['failures'].get('forbidden',False) for r in rows),strict_clean=sum(clean(r) and r['strict_transfer'] for r in rows))

def filtered(rows,strict=False):return [dict(r,success=bool(clean(r) and (not strict or r['strict_transfer']))) for r in rows]

def sources(folder,protocol):
    m=read(folder/'manifest.json')
    for name,h in m['sources'].items():assert protocol['hashes'].get(name)==h,(folder,name)
    for name,h in m['bank_hashes'].items():assert sha(Path(m['args']['bank'])/name)==h
    for f in (folder/'source').iterdir():
        key=str(ROOT/'scripts'/f.name)
        if key in protocol['hashes']:assert sha(f)==protocol['hashes'][key],f

def verify(study):
    protocol=read(study/'protocol.json');assert sha(study/'protocol.json')==(study/'protocol.sha256').read_text().strip()
    check(protocol['hashes']);check(protocol['prior_hashes'])
    out=dict(passed=False,sources=len(protocol['hashes']),protected=len(protocol['prior_hashes']),training={},selections={},evaluations={},comparisons={},strict_comparisons={})
    for seed in protocol['seeds']:
        f=study/f'seed_{seed}';train=f/'train';c=read(train/'feedback_contract.json');sources(train,protocol)
        assert c['local_updates']==16 and c['active']==448000 and c['warmup']==10500 and c['total']==458500 and c['start_s']==10.
        assert c['initial_settling_excluded']==700 and c['decisions']==8960 and c['feedback_window_ticks']==224000 and c['teacher_prefix_ticks']==224000
        assert equal(load(train/'initial.pt'),load(DATA/'preflight_01'/f'seed_{seed}/smoke/initial.pt'))
        training=validate(train);tr=np.load(train/'trace.npz');start=np.load(train/'startup.npz');names=read(train/'results.json')['joint_names'];fingers=[i for i,n in enumerate(names) if any(t in n for t in ['_zero_','_one_','_two_','_three_','_four_','_five_','_six_'])]
        assert len(fingers)==14 and np.array_equal(tr['command'][:,:,fingers],np.broadcast_to(start['command'][None,:,fingers],tr['command'][:,:,fingers].shape))
        out['training'][str(seed)]=training
        baseline=DATA/'preflight_01'/f'seed_{seed}/calibration';candidates=read(f/'development_candidates.json');assert [x['update'] for x in candidates]==[4,8,12,16]
        for row in candidates:
            p=Path(row['evaluation']);sources(p,protocol);assert row['gate']==gate(p,baseline);validate(p);matched(baseline,p,start=10.)
            assert sha(row['checkpoint'])==row['sha256']==read(p/'manifest.json')['feedback_checkpoint_sha256']
        passing=[r for r in candidates if r['gate']['passed']]
        selected=read(f/'selection.json');chosen=selected['selected']
        if passing:
            best=min(passing,key=lambda r:(-r['gate']['clean'],r['gate']['physical'],r['gate']['effort']['metrics_whole_trial']['ratio'],r['start_s']))
            earlier=read(f/'prospective8.json');assert earlier['checkpoint']==best['checkpoint'] and earlier['gate']==gate(Path(earlier['evaluation']),baseline)
            validate(Path(earlier['evaluation']));matched(baseline,Path(earlier['evaluation']),start=8.)
            assert chosen==(earlier if earlier['gate']['passed'] else best)
        else:assert chosen['parent_fallback']
        assert selected['final_sha256']==sha(train/'final.pt') and selected['all16_updates_trained_at_10s'] and not selected['holdout_used']
        barrier=read(f/'holdout_barrier.json')
        for other in protocol['seeds']:
            selection=study/f'seed_{other}/selection.json';assert sha(selection)==barrier['selections'][str(selection)]
            assert read(selection)['selected_at']<barrier['timestamp'] and (study/f'seed_{other}/train/final.pt').stat().st_mtime<barrier['timestamp']
        out['selections'][str(seed)]=selected
        for bank in ['development_old','development_transfer','holdout']:
            distinct_selected=not chosen.get('parent_fallback') and not (chosen['update']==16 and chosen['start_s']==10.)
            controllers=['reference','parent','current_61111','current_61117','final']+(['selected'] if distinct_selected else [])
            allrows={};geometry=None
            for label in controllers:
                p=f/f'{bank}_{label}';sources(p,protocol);checks,rows,g=evaluation(p,24);geometry=g
                outcome=physical(p,require_clean=False);outcome.update(summary(rows));outcome['checks']=checks
                window=6. if label.startswith('current') else (chosen['start_s'] if label=='selected' else 10.)
                outcome['matched']=matched(f/f'{bank}_reference',p,start=window)
                if label in ['final','selected']:
                    validate(p);cp=selected['final_checkpoint'] if label=='final' else chosen['checkpoint'];assert read(p/'manifest.json')['feedback_checkpoint_sha256']==sha(cp)
                elif label.startswith('current'):
                    old_seed=int(label.split('_')[-1]);actors(p,OLD/'ppo_01'/f'seed_{old_seed}/block_03/train/final.pt')
                if bank=='holdout':assert (p/'manifest.json').stat().st_mtime>barrier['timestamp']
                outcome['by_condition']=[dict(case=r['case'],height=g[r['geometry_index']]['height'],slope=g[r['geometry_index']]['pitch_deg'],**summary([r])) for r in rows]
                out['evaluations'][f'{seed}_{bank}_{label}']=outcome;allrows[label]=rows
            for label in ['final']+(['selected'] if distinct_selected else []):
                key=f'{seed}_{bank}_{label}';out['comparisons'][key]={};out['strict_comparisons'][key]={}
                for base in ['reference','parent','current_61111','current_61117']:
                    out['comparisons'][key][base]=compare(filtered(allrows[label]),filtered(allrows[base]),geometry)
                    out['strict_comparisons'][key][base]=compare(filtered(allrows[label],strict=True),filtered(allrows[base]),geometry)
    assert sum(x['active']+x['warmup'] for x in out['training'].values())==917000
    check(protocol['hashes']);check(protocol['prior_hashes']);out['passed']=True
    out['scope']='All16updates trained10–20s continuous bounded reference-assisted feedback. Any8s selection is an untrained prospective evaluation, not8s training. Known failing cases were training targets; all outcomes remain reported. Fresh24heldout cases use12 rigid geometries twice. Two seeds; torque is a proxy, not electrical energy/global optimum. No policy promotion.'
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    write(a.output/'verification.json',verify(a.study));shutil.copy2(__file__,a.output/'verification_source.py');print('VERIFIED',a.output)
