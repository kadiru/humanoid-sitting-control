"""Run the same twelve paired starts with the original contact and action checks."""
import argparse
import csv
from pathlib import Path
import shutil
import numpy as np

from controlled_seat_step_study import measure
from run import P, PREPARED, DISK_RESERVE, read, sha, stage, write
from rescan_sitting import points, shoe_frame

BASE=P/'controlled_seat_step_01'
STUDY=P/'post_step_rescan_01'
DEMOS={'Chair_Wire':(0,4),'Anora':(16,5),'Basic':(16,6)}


def prepare():
    from stance23_planner import model
    STUDY.mkdir(exist_ok=True)
    library={}
    demos=[]
    for name,(half,i) in DEMOS.items():
        p=P/f'protected_hip_landing_{half}'/'evaluation'
        audit=read(p/'independent_audit.json')
        episode=read(p/'results.json')
        assert audit['passed'] and sha(p/'trajectory.npz')==audit['files']['trajectory.npz']
        assert episode['episodes'][i]['id']=='sit/'+name
        body_names=episode['body_names']
        ids=[body_names.index(n) for n in ['pelvis','left_ankle_roll_link','right_ankle_roll_link']]
        with np.load(p/'standing_scan.npz') as s, np.load(p/'trajectory.npz') as data:
            assert data['good'][-250:,i].all()
            _,mid,axes=shoe_frame(model,s['q'][0,i],s['gravity'][0,i])
            rotation=s['rotation'][0,i]
            body=data['body'][:,i,ids,:].astype(float)
            body[...,:3]-=episode['origins'][i]
            weight=data['physical_bodyweight_N'][:,i]
            load=data['physical_force_w'][:,i,:5,2].sum(1)/weight
            contact=int(np.flatnonzero(load>.05)[0])
            library.update({name+'_'+k:v for k,v in dict(
                origin=s['root'][0,i]+rotation@mid,frame=rotation@axes,
                cloud=points(s['depth'][0,i],s['intrinsics'][0,i],s['mount'][0],s['q'][0,i],s['gravity'][0,i],model),
                body=body,q=data['q'][:,i],desired=data['desired'][:,i],force=data['physical_force_w'][:,i,5:],loaded=contact).items()})
        demos.append(dict(asset=name,run=str(p),index=i,trace_sha256=audit['files']['trajectory.npz'],
                          scan_sha256=sha(p/'standing_scan.npz'),contact_tick=contact))
    file=STUDY/'rescan_force_library.npz'
    with file.open('xb') as stream:
        np.savez_compressed(stream,**library)
    write(STUDY/'library_inputs.json',demos)


def run(name,with_rescan,shift=0.,gain=.25,balance=0.,pitch=0.,wire=False,wire_hold=False,wire_pressure=False):
    original=BASE/('with_settle' if with_rescan else 'without')
    r=read(BASE/'with_settle/evaluation.freeze.json')
    for path,hash_ in r['inputs'].items():
        assert sha(path)==hash_,path
    target=STUDY/name
    target.mkdir()
    bank=target/'bank'
    shutil.copytree(original/'bank',bank)
    shutil.copyfile(original/'landing_profile.json',target/'landing_profile.json')
    config=read(original/'step_config.json')
    if with_rescan:
        config.update(rescan_library=str(STUDY/'rescan_force_library.npz'),seat_shift_m=shift,tracking_gain=gain,
                      clearance_pitch=pitch,balance_strength=balance,balance_feedback_assets=['Chair_Wire','Anora'] if balance else [])
    if wire and with_rescan:
        config['wire_clearance']=dict(fore_aft_m=.013,roll_rad=-.018)
        config['wire_hold']=wire_hold
        config['wire_pressure_hold']=wire_pressure
    write(target/'step_config.json',config)
    write(target/'reservation.json',dict(files={str(f):sha(f) for f in bank.iterdir()}))
    cmd=r['command']
    more=[P/'rescan_sitting_study.py',target/'landing_profile.json',target/'step_config.json',Path(config['steps'])]
    if with_rescan:
        more.extend([P/'rescan_sitting.py',STUDY/'rescan_force_library.npz',STUDY/'library_inputs.json'])
    stage(target,'evaluation',bank,PREPARED/'library.npz',
          worker=P/('rescan_sitting_worker.py' if with_rescan else 'controlled_seat_step_worker.py'),
          extra=cmd[3:cmd.index('--mode')],inputs=[*map(Path,r['inputs']),*more],disk_reserve=DISK_RESERVE)


def score(name):
    from publish_confirmed import assess
    d=STUDY/name
    ev=d/'evaluation'
    step=measure(d)
    sits,_=assess(d)
    receipt=read(ev/'rescan_plan.json') if (ev/'rescan_plan.json').exists() else []
    admitted={r['row']:r['admitted'] for r in receipt}
    results=[]
    with np.load(ev/'trajectory.npz') as z, np.load(ev/'contact_attribution.npz') as c:
        hip=[i for i,n in enumerate(c['probe_links']) if 'hip_pitch' in n or 'hip_roll' in n]
        assert len(hip)==4
        peak=c['chair_history_peak_N'][:,:,hip].max(axis=(0,2))
        f=z['physical_force_w']
        bw=z['physical_bodyweight_N']
        alive=z['learn']
        support=np.abs(f[:,:,5:,2])/bw[:,:,None]
        w=np.abs(z['physical_torque']*z['physical_joint_velocity']).sum(-1)
        torque=np.sqrt(np.square(z['physical_torque']).mean(-1))
        cmd_error=np.max(np.abs(z['command'][:,:,:23]-z['previous'][:,:,:23]))
        assert cmd_error<.0030003
        names=read(ev/'results.json')['body_names']
        shoes=z['body'][:,:,[names.index(k) for k in ('left_ankle_roll_link','right_ankle_roll_link')],:3]
        for i,(s,r) in enumerate(zip(step,sits,strict=True)):
            held=r['terminal_hold_s']
            k=max(1,int(round(held/.02)))
            full=s['clean_prestage'] and (not s['steps'] or s['step_verified']) and r['supported_sit']
            minimal_foot_ratio=float(support[:,i][alive[:,i]].min())
            results.append(dict(mode=name,case=s['id'],step=bool(s['steps']),step_verified=s['step_verified'],
                retargeted=admitted.get(i,False),supported=bool(full),strict=bool(full and r['strict_clean']),
                hold_s=held,longest_hold_s=r['longest_hold_s'],load_pct=r['terminal_seat_weight_pct'],
                hold_power_W=round(float(w[-k:,i].mean()),3) if held else '',
                hold_torque_RMS_Nm=round(float(torque[-k:,i].mean()),3) if held else '',
                descent_mechanical_work_J=round(float(w[alive[:,i],i].sum()*.02),3),
                step_mechanical_work_J=round(s['alignment_effort_J'],3),foot_slip=bool(r['loaded_foot_slip']),
                min_foot_load_pct=round(100*minimal_foot_ratio,2),
                foot_movement_cm=round(100*float(np.linalg.norm(shoes[alive[:,i],i,:,:2]-shoes[0,i,:,:2],axis=-1).max()),2),
                hip_peak_N=round(float(max(peak[i],s['pre_hip_N'])),2),reason=r['reason'],contact=r['contact_detail']))
    with (d/'case_results.csv').open('w',newline='') as f:
        a=csv.DictWriter(f,fieldnames=results[0].keys())
        a.writeheader()
        a.writerows(results)
    print(name,'supported',sum(r['supported'] for r in results),'strict',sum(r['strict'] for r in results),
          [(r['case'],r['supported'],r['reason'],r['contact'],r['hold_s'],r['min_foot_load_pct'])
           for r in results if r['step'] or r['supported']],flush=True)
    return results


def paired(candidate):
    a,b=score('without'),score(candidate)
    assert len(a)==len(b)==12
    for name in ('bank.npz','metadata.json','geometry.json'):
        assert sha(STUDY/'without/bank'/name)==sha(STUDY/candidate/'bank'/name)
    for x,y in zip(a,b,strict=True):
        assert x['case']==y['case']
        y['baseline_pass']=x['supported']
        y['repaired']=y['supported'] and not x['supported']
        y['regressed']=x['supported'] and not y['supported']
    summary=dict(before=sum(r['supported'] for r in a),after=sum(r['supported'] for r in b),
                 repaired=[r['case'] for r in b if r['repaired']],regressed=[r['case'] for r in b if r['regressed']])
    write(STUDY/'paired_summary.json',summary)
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('operation',choices=('prepare','run','score','paired'))
    p.add_argument('--name',default='retarget')
    p.add_argument('--shift',type=float,default=0.)
    p.add_argument('--wire-pressure',action='store_true')
    p.add_argument('--wire-hold',action='store_true')
    p.add_argument('--wire',action='store_true')
    p.add_argument('--pitch',type=float,default=0.)
    p.add_argument('--balance',type=float,default=0.)
    p.add_argument('--gain',type=float,default=.25)
    a=p.parse_args()
    if a.operation=='prepare':prepare()
    if a.operation=='run':
        run(a.name,a.name!='without',a.shift,a.gain,a.balance,a.pitch,a.wire,a.wire_hold,a.wire_pressure)
        score(a.name)
    if a.operation=='score':score(a.name)
    if a.operation=='paired':paired(a.name)
