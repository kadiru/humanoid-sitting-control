"""Compare the new settled posture with the four-pass controller on the same pairs."""
import argparse
import csv
from pathlib import Path
import shutil

import numpy as np

from run import DISK_RESERVE, P, PREPARED, read, sha, stage, write
import rescan_sitting_study

BASE=P/'post_step_rescan_01'
STUDY=P/'sitting_load_effort_01'
SOURCE=('sitting_load_effort.py','sitting_load_effort_worker.py','sitting_load_effort_study.py')


def prepare():
    STUDY.mkdir()
    ev=BASE/'without/evaluation'
    result=read(ev/'results.json')
    audit=read(ev/'independent_audit.json')
    assert audit['passed'] and audit['strict_predicates_exact'] and not audit['physical_tolerances_changed']
    assert sha(ev/'trajectory.npz')==audit['files']['trajectory.npz']
    i=next(i for i,r in enumerate(result['episodes']) if r['id']=='sit/Basic/fore_aft')
    ids=[result['body_names'].index(n) for n in ['pelvis','left_ankle_roll_link','right_ankle_roll_link']]
    with np.load(ev/'trajectory.npz') as z:
        assert z['good'][-250:,i].all()
        b=z['body'][:,i,ids].copy()
        b[...,:3]-=np.array(result['origins'][i])
        np.savez_compressed(STUDY/'efficient_basic.npz',hold_pose=np.median(b[-90:],axis=0),
                            start_feet=b[0,1:],q=np.median(z['q'][-90:,i],axis=0),
                            command=np.median(z['command'][-90:,i,:23],axis=0),
                            floor_force=z['physical_force_w'][-90:,i,5:].mean(0))
    write(STUDY/'reference.json',dict(run=str(ev),trace_sha256=audit['files']['trajectory.npz'],
                                     case=result['episodes'][i]['id'],tail_samples=90))


def launch(name,phase):
    source=BASE/'pressure_hold'
    frozen=read(source/'evaluation.freeze.json')
    for path,digest in frozen['inputs'].items():
        assert sha(path)==digest,path
    target=STUDY/name
    target.mkdir()
    bank=target/'bank'
    shutil.copytree(source/'bank',bank)
    shutil.copyfile(source/'landing_profile.json',target/'landing_profile.json')
    config=read(source/'step_config.json')
    if phase is not None:
        config['rest_refinement']={**phase,'inputs':str(STUDY/'efficient_basic.npz')}
    write(target/'step_config.json',config)
    write(target/'reservation.json',dict(files={str(f):sha(f) for f in bank.iterdir()}))
    cmd=frozen['command']
    inputs=[*map(Path,frozen['inputs']),target/'step_config.json',target/'landing_profile.json']
    if phase is not None:
        inputs.extend([*(P/n for n in SOURCE),STUDY/'efficient_basic.npz',STUDY/'reference.json'])
    stage(target,'evaluation',bank,PREPARED/'library.npz',
          worker=P/('sitting_load_effort_worker.py' if phase is not None else 'rescan_sitting_worker.py'),
          extra=cmd[3:cmd.index('--mode')],inputs=inputs,disk_reserve=DISK_RESERVE)
    score(name)


def score(name):
    rescan_sitting_study.STUDY=STUDY
    results=rescan_sitting_study.score(name)
    folder=STUDY/name/'evaluation'
    with np.load(folder/'trajectory.npz') as z:
        load=(z['physical_force_w'][-250:,:,:5]*z['physical_normal_w'][-250:,:,:5]).sum(-1).clip(0).sum(-1)
        load/=z['physical_bodyweight_N'][-250:]
        power=np.abs(z['physical_torque'][-250:,:,:23]*z['physical_joint_velocity'][-250:,:,:23])
        torque=np.sqrt(np.square(z['physical_torque'][-250:,:,:23]).mean(-1))
        foot_normal=z['physical_force_w'][:,:,5:,2]/z['physical_bodyweight_N'][:,:,None]
        feet_slip=z['physical_slip_m_s'][:,:,5:]
        archive=folder/'rest_refinement.npz'
        traces={}
        if archive.exists():
            with np.load(archive,allow_pickle=False) as ref:
                traces={int(row):ref[row] for row in ref.files if len(ref[row])}
        for i,r in enumerate(results):
            r['hold_seat_normal_pct']=round(float(load[:,i].mean())*100,2) if r['supported'] else None
            r['hold_mechanical_power_W']=round(float(power[:,i].sum(1).mean()),3) if r['supported'] else None
            r['hold_body_joint_RMS_Nm']=round(float(torque[:,i].mean()),3) if r['supported'] else None
            if i in traces:
                start=int(traces[i][0,0])
                r['refinement_start_tick']=start
                r['refinement_min_foot_load_pct']=round(float(foot_normal[start:,i].min())*100,2)
                r['refinement_max_foot_slip_m_s']=round(float(feet_slip[start:,i].max()),5)
                r['refinement_feet_planted']=bool(
                    (foot_normal[start:,i]>.05).all() and (feet_slip[start:,i]<=.02).all())
            for key,value in r.items():
                if isinstance(value,np.generic):
                    r[key]=value.item()
    write(STUDY/name/'measures.json',results)
    print('targets:',[r for r in results if r['case'] in ('sit/Anora/fore_aft','sit/Basic/fore_aft')],flush=True)
    return results


def paired(name):
    baseline=read(STUDY/'baseline/measures.json')
    candidate=read(STUDY/name/'measures.json')
    before=STUDY/'baseline/bank'
    after=STUDY/name/'bank'
    for n in ('geometry.json','metadata.json','bank.npz'):
        assert sha(before/n)==sha(after/n)==sha(BASE/'pressure_hold/bank'/n)
    assert len(baseline)==len(candidate)==12
    protected={r['case'] for r in csv.DictReader((BASE/'pressure_hold/case_results.csv').open()) if r['supported']=='True'}
    assert len(protected)==4
    for a,b in zip(baseline,candidate,strict=True):
        assert a['case']==b['case']
        b['baseline_pass']=a['supported']
        b['protected_pass']=a['case'] in protected
    combined=baseline+candidate
    with (STUDY/'paired_hold_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in combined for k in r)))
        w.writeheader()
        w.writerows(combined)
    first={r['case']:r for r in baseline}
    last={r['case']:r for r in candidate}
    anora='sit/Anora/fore_aft'
    basic='sit/Basic/fore_aft'
    summary=dict(before=sum(r['supported'] for r in baseline),after=sum(r['supported'] for r in candidate),
                 protected_passes=sorted(protected),
                 lost_protected=sorted(n for n in protected if not last[n]['supported']),
                 repairs=[x for x in last if last[x]['supported'] and not first[x]['supported']],
                 lost_pair_passes=[x for x in last if not last[x]['supported'] and first[x]['supported']],
                 anora=[first[anora]['hold_seat_normal_pct'],last[anora]['hold_seat_normal_pct']],
                 basic=[first[basic]['hold_mechanical_power_W'],last[basic]['hold_mechanical_power_W']])
    summary['accepted']=bool(
        not summary['lost_protected'] and not summary['lost_pair_passes']
        and all(last[case].get('refinement_feet_planted') for case in (anora,basic))
        and None not in summary['anora'] and summary['anora'][1]>summary['anora'][0]
        and None not in summary['basic'] and summary['basic'][1]<summary['basic'][0])
    write(STUDY/'summary.json',summary)
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','score','paired'])
    p.add_argument('--name',default='low_shear')
    p.add_argument('--load',type=float,default=.15)
    p.add_argument('--depth',type=float,default=.0022)
    p.add_argument('--roll',type=float,default=.018)
    p.add_argument('--setback',type=float,default=-.006)
    p.add_argument('--basic-share',type=float,default=.15)
    p.add_argument('--basic-blend-ticks',type=int,default=154)
    a=p.parse_args()
    if a.command=='prepare':prepare()
    if a.command=='run':
        phase=dict(start_tick=637,blend_ticks=126,anora_load_target=a.load,anora_drop_limit_m=a.depth,
                   anora_roll_pitch_rad=[a.roll,-.022],basic_setback_m=a.setback,
                   basic_share=a.basic_share,basic_blend_ticks=a.basic_blend_ticks)
        launch(a.name,None if a.name=='baseline' else phase)
    if a.command=='score':score(a.name)
    if a.command=='paired':paired(a.name)
