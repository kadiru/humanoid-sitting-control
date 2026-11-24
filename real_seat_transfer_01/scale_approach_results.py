"""Export verified demonstrations, matched outcomes and the 4090 optimizer receipts."""
import zipfile
from pathlib import Path
import numpy as np
from scale_approach_study import STUDY, SUITE, P, read, sha, rows, csvout, write


def main():
    final=read(STUDY/'final.json')
    assert sha(final['checkpoint'])==final['sha256'] and sha(final['library'])==final['library_sha256']
    baseline={r['case']:r for r in rows(SUITE/'case_results.csv')}
    collection=read(STUDY/'collected_demos.json')
    demos={}
    for r in collection:
        assert baseline[r['case']]['split']=='train'
        demos.setdefault(r['folder'],[]).append(r)
    names=[]
    observation=[]
    command=[]
    for folder,group in demos.items():
        directory=Path(folder)
        cs=read(directory/'scores.json')
        with np.load(directory/'evaluation/trajectory.npz') as z:
            indices=[r['index'] for r in group]
            assert z['good'][-250:,indices].all()
            for r in group:
                s=cs[r['index']]
                assert s['supported_sit'] and s['unconfirmed_force_samples']==0 and s['case']==r['case']
                names.append(r['case'])
            observation.extend(z['obs'][:,indices].transpose(1,0,2))
            command.extend(z['command'][:,indices,:23].transpose(1,0,2))
    if names:
        np.savez_compressed(STUDY/'collected_sits.npz',case=names,obs=np.asarray(observation).reshape(-1,447),
                            command_rad=np.asarray(command).reshape(-1,23),episode_offsets=np.arange(len(names)+1)*1000,
                            control_dt_s=np.asarray(.02))
    paired=rows(STUDY/'paired_results.csv')
    assert len(paired)==128
    summary=read(STUDY/'paired_summary.json')
    episodes={}
    for r in paired:
        b=baseline[r['case']]
        p=Path(b['trace']).parent
        q=STUDY/r['evaluation_folder']/'evaluation'
        if q not in episodes:
            for name in ('bank.npz','geometry.json','metadata.json'):
                assert sha(p.parent/'bank'/name)==sha(q.parent/'bank'/name)
            before_manifest,after_manifest=(read(d/'manifest.json') for d in (p,q))
            assert before_manifest['args']['seed']==after_manifest['args']['seed']
            assert after_manifest['action_limits_unchanged'] and after_manifest['physical_criteria_unchanged']
            audit=read(q/'independent_audit.json')
            assert audit['passed'] and audit['strict_predicates_exact'] and not audit['physical_tolerances_changed']
        for ev in (p,q):
            if ev not in episodes:
                episodes[ev]=read(ev/'results.json')['episodes']
        i=int(r['trajectory_index'])
        old,new=episodes[p][i],episodes[q][i]
        assert old['id']==new['id'] and old['steps']==new['steps']==1000
        flags_before,flags_after=old['physical_failures'],new['physical_failures']
        r.update(baseline_reason=b['outcome'],baseline_contact=b['contact'],
                 before_forbidden_guard=flags_before[1],after_forbidden_guard=flags_after[1],
                 new_forbidden_guard=flags_after[1] and not flags_before[1],
                 new_hard_guards=[j for j,(x,y) in enumerate(zip(flags_before,flags_after)) if y and not x],
                 lost_strict=r['baseline_strict']=='True' and r['strict_clean']=='False')
    csvout(STUDY/'case_details.csv',paired)
    summary['strict_regressions']=[r['case'] for r in paired if r['lost_strict']]
    summary['new_forbidden_guard']=[r['case'] for r in paired if r['new_forbidden_guard']]
    summary['passing_cases_with_unconfirmed_force_samples']=[r['case'] for r in paired
        if r['supported_sit']=='True' and int(r['unconfirmed_force_samples'])>0]
    training_changes=[]
    for size in (88,64):
        ids=[case for case in names if int(baseline[case]['scale_pct'])==size]
        training_changes.append(dict(scale=size,verified_cases=len(ids),new_cases=[n for n in ids if baseline[n]['supported_sit']=='False']))
    summary['collection']=training_changes
    table=[]
    for size in (100,88,76,64):
        rr=[r for r in paired if int(r['scale_pct'])==size]
        table.append(dict(scale=size,trials=len(rr),before=sum(r['baseline_supported']=='True' for r in rr),
                          after=sum(r['supported_sit']=='True' for r in rr),repairs=sum(r['repaired']=='True' for r in rr),
                          regressions=sum(r['regressed']=='True' for r in rr),matches=sum(r['camera_matched']=='True' for r in rr)))
    summary['scale_results']=table
    write(STUDY/'result_summary.json',summary)
    with zipfile.ZipFile(STUDY/'scale_conditioned_policy.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ('final.json','train_library.npz','policy_'+str(final['updates'])+'.pt','updates.jsonl',
                     'result_summary.json','case_details.csv','verified_collection.csv'):
            z.write(STUDY/name,name)
        for name in ('scale_approach.py','scale_approach_worker.py','scale_approach_study.py'):
            z.write(P/name,name)
        for d in sorted(STUDY.glob('test_*')):
            for name in ('evaluation.freeze.json','witness_counts.json','scores.json','evaluation/approach_choice.npz','evaluation/independent_audit.json'):
                z.write(d/name,d.name+'/'+name)
    print(summary,flush=True)


if __name__=='__main__':
    main()
