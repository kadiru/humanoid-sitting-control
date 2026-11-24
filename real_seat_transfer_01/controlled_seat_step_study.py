"""Verify the single step before rerunning the exact twelve paired sitting starts."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation

from run import P, PREPARED, DISK_RESERVE, read, write, sha, stage
from controlled_seat_step import compile_paths, BASE_END


SOURCE=P/'rear_camera_alignment_01'
STUDY=P/'controlled_seat_step_01'
CODE=('controlled_seat_step.py','controlled_seat_step_study.py','controlled_seat_step_worker.py',
      'seat_alignment.py','stance23_planner.py')


def pilot_bank(bank,variants):
    source=P/'compact_maple/bank'
    g=read(source/'geometry.json')[0]
    meta=read(source/'metadata.json')
    ids=[f"sit/{g['name']}/step_{a}_cut_{cut}" for a,cut in variants]
    bank.mkdir()
    write(bank/'geometry.json',[dict(g,case=i,surface_id=i,label=k) for i,k in enumerate(ids)])
    write(bank/'metadata.json',dict(meta,cases=len(ids),ids=ids,evaluation_split='step_pilot'))
    with np.load(source/'bank.npz') as z:
        data={k:np.repeat(z[k],len(ids),axis=int(k=='q_post')) for k in z.files if k!='reference'}
    np.savez_compressed(bank/'bank.npz',reference=np.arange(len(ids)),**data)


def run(name,config,variants=None):
    dest=STUDY/name
    dest.mkdir()
    old=SOURCE/'without'
    bank=dest/'bank'
    frozen=read(old/'evaluation.freeze.json')
    for path,h in frozen['inputs'].items():
        # The previous alignment worker and its orchestration are superseded
        # only for this pre-stage. The descent dependency hashes must agree.
        if Path(path).name not in ('seat_alignment_worker.py','seat_alignment_steps.py','seat_alignment_study.py'):
            assert sha(path)==h,path
    shutil.copyfile(old/'landing_profile.json',dest/'landing_profile.json')
    if variants is None:
        shutil.copytree(SOURCE/'paired_bank',bank)
    else:
        pilot_bank(bank,variants)
    config=dict(config,steps=str(STUDY/('commands_feedback.npz' if config.get('feedback') else 'commands.npz')),templates=str(SOURCE/'templates.npz'),
                weights=str(SOURCE/'alignment.pt'))
    write(dest/'step_config.json',config)
    write(dest/'reservation.json',dict(files={str(f):sha(f) for f in bank.iterdir()}))
    args=frozen['command']
    inputs=[Path(p) for p in frozen['inputs'] if Path(p).name not in
            ('seat_alignment_worker.py','seat_alignment_steps.py','seat_alignment_study.py')]
    inputs.extend([P/f for f in CODE])
    inputs.extend(Path(config[k]) for k in ('steps','templates','weights'))
    inputs.extend([dest/'step_config.json',dest/'landing_profile.json'])
    inputs.extend(Path(g['mesh_file']) for g in read(bank/'geometry.json'))
    stage(dest,'evaluation',bank,PREPARED/'library.npz',worker=P/'controlled_seat_step_worker.py',
          extra=args[3:args.index('--mode')],inputs=inputs,disk_reserve=DISK_RESERVE)


def measure(folder):
    ev=folder/'evaluation'
    rs=read(ev/'results.json')
    assert read(ev/'manifest.json')['action_limits_unchanged']
    audit=read(ev/'independent_audit.json')
    assert audit['passed'] and audit['strict_predicates_exact'] and not audit['physical_tolerances_changed']
    assert audit['files']['trajectory.npz']==sha(ev/'trajectory.npz')
    with np.load(ev/'single_step.npz') as d:
        x={k:d[k] for k in d.files}
    z=x['body']
    names=x['names'].tolist()
    ix=[names.index(s+'_ankle_roll_link') for s in ('left','right')]
    shoe=z[:,:,ix,:3]
    disp=shoe-x['initial_body'][None,:,ix,:3]
    lifted=x['clearance']-x['initial_clearance']
    obs=[]
    for i,r in enumerate(rs['episodes']):
        airborne=np.flatnonzero(lifted[:,i,0]>.002)
        t=int(airborne[0]) if len(airborne) else None
        grounded=np.flatnonzero((x['floor_load'][:,i,0]<.01)&(x['floor_load'][:,i,1]>.75))
        un=int(grounded[0]) if len(grounded) else None
        stable=x['floor_load'][-50:,i].min()>.15 and x['flat_speed'][-50:,i].max()<.02
        retained=bool(np.linalg.norm(shoe[-1,i,0,:2]-shoe[-50,i,0,:2])<.002 and
                      np.linalg.norm(disp[-1,i,1,:2])<.002)
        had_slip=bool(x['slip'][:,i].any())
        witness=bool(x['verified_floor'][:,i].all())
        clean=bool(not x['flags'][:,i].any() and not had_slip and witness)
        moved=(np.linalg.norm(disp[-1,i,0,:2])>.012 or
               (Rotation.from_quat(z[-1,i,ix[0],3:])*Rotation.from_quat(x['initial_body'][i,ix[0],3:]).inv()).magnitude()>.06)
        gate=bool(t is not None and t>=3 and (x['floor_load'][t-3:t,i,0]<.01).all()
                  and (x['floor_load'][t-3:t,i,1]>.75).all())
        got=bool(gate and moved and stable and retained and clean)
        obs.append(dict(id=r['id'],steps=int(x['options'][i]),cutoff=int(x['cutoffs'][i]),step_verified=got,
                        clean_prestage=clean,physical_fault=bool(x['flags'][:,i].any()),loaded_foot_slip=had_slip,
                        floor_witness=witness,first_unloaded_s='' if un is None else (un+1)*.02,
                        first_lift_s='' if t is None else (t+1)*.02,
                        load_before_lift_pct='' if t is None else float(x['floor_load'][max(0,t-3):t+1,i,0].max()*100),
                        pelvis_dx_cm=float((x['body'][-1,i,0,0]-x['initial_body'][i,0,0])*100),
                        pelvis_dy_cm=float((x['body'][-1,i,0,1]-x['initial_body'][i,0,1])*100),
                        pelvis_yaw_deg=float((Rotation.from_quat(x['body'][-1,i,0,3:])*
                            Rotation.from_quat(x['initial_body'][i,0,3:]).inv()).as_euler('zyx',degrees=True)[0]),
                        max_sole_lift_mm=float(lifted[:,i,0].max()*1000),
                        left_dx_cm=float(disp[-1,i,0,0]*100),left_dy_cm=float(disp[-1,i,0,1]*100),
                        right_travel_cm=float(np.linalg.norm(disp[-1,i,1,:2])*100),
                        settled_left_pct=float(x['floor_load'][-50:,i,0].mean()*100),
                        settled_right_pct=float(x['floor_load'][-50:,i,1].mean()*100),
                        both_feet_settled=bool(stable and retained),
                        max_loaded_foot_cm_s=float(np.where(x['floor_load'][:,i]>.01,x['floor_speed'][:,i],0).max()*100),
                        alignment_effort_J=float(x['prestage_power_W'][:,i].sum()*.02),
                        pre_hip_N=float(x['hip_N'][:,i].max())))
    return obs


def save(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def verify_pilot(name):
    scores=measure(STUDY/name)
    save(STUDY/(name+'_step_checks.csv'),scores)
    print(name,[(r['steps'],r['cutoff'],r['step_verified'],round(r['max_sole_lift_mm'],1),
                round(r['left_dx_cm'],2),round(r['left_dy_cm'],2),round(r['settled_left_pct'],1),
                round(r['max_loaded_foot_cm_s'],2),r['first_unloaded_s'],r['first_lift_s'],
                r['physical_fault'],r['floor_witness']) for r in scores],flush=True)
    return scores


def prepare(feedback=False):
    STUDY.mkdir(exist_ok=True)
    dest=STUDY/('commands_feedback.npz' if feedback else 'commands.npz')
    if not dest.exists():
        compile_paths(dest)


def pilot(name,variants):
    run(name,dict(mode='pilot',choices=[a for a,c in variants],cutoffs=[c for a,c in variants]),variants)
    verify_pilot(name)


def pairs(cut,options,feedback=0.,taper=False):
    calibration=[r for f in sorted(STUDY.glob('pilot*/evaluation.score.json')) for r in measure(f.parent)
                 if r['cutoff']==cut and r['step_verified']]
    assert options
    displacements=[]
    for a in options:
        samples=[r for r in calibration if r['steps']==a]
        assert samples
        displacements.append(np.median([[r['pelvis_dx_cm']/100,r['pelvis_dy_cm']/100,
                                         np.deg2rad(r['pelvis_yaw_deg'])] for r in samples],axis=0).tolist())
    cases=read(SOURCE/'paired_cases.json')
    assert len(cases)==12
    aligned='with_settle' if taper else ('with_feedback' if feedback else 'with')
    for name,mode in [('without','idle'),(aligned,'aligned')]:
        if not (STUDY/name/'evaluation.score.json').exists():
            run(name,dict(mode=mode,validated_choices=options,validated_displacements=displacements,cutoff=cut,feedback=feedback if mode=='aligned' else 0.,settle_taper=bool(taper),templates_by_row=[c['asset'] for c in cases]))
    report(aligned)


def report(aligned='with'):
    from publish_confirmed import assess
    rows=[]
    for mode in ('without',aligned):
        folder=STUDY/mode
        for file in ('bank.npz','geometry.json','metadata.json'):
            assert sha(folder/'bank'/file)==sha(SOURCE/'paired_bank'/file)
        ev=folder/'evaluation'
        step=measure(folder)
        scored,_=assess(folder)
        with np.load(ev/'trajectory.npz') as z, np.load(ev/'contact_attribution.npz') as a:
            probe=a['probe_links'].tolist()
            hips=[i for i,n in enumerate(probe) if 'hip_pitch' in n or 'hip_roll' in n]
            assert len(hips)==4
            hip=a['chair_history_peak_N'][:,:,hips].max(axis=(0,2))
            power=np.abs(z['physical_torque']*z['physical_joint_velocity']).sum(-1)
            torque=np.sqrt(np.square(z['physical_torque']).mean(-1))
            good=z['good'][-250:].all(0)
            for i,(s,r) in enumerate(zip(step,scored,strict=True)):
                complete=s['clean_prestage'] and (not s['steps'] or s['step_verified'])
                rows.append(dict(mode=mode,**s,seat_hold_s=r['terminal_hold_s'],supported=bool(complete and r['supported_sit']),
                                 strict=bool(complete and r['strict_clean']),descent_supported=r['supported_sit'],
                                 load_pct=r['terminal_seat_weight_pct'],hold_power_W=float(power[-250:,i].mean()) if good[i] else '',
                                 hold_torque_RMS_Nm=float(torque[-250:,i].mean()) if good[i] else '',
                                 hip_housing_peak_N=max(float(hip[i]),s['pre_hip_N']),reason=r['reason']))
    save(STUDY/('paired_'+aligned+'.csv'),rows)
    for m in ('without',aligned):
        r=[x for x in rows if x['mode']==m]
        print(m,len(r),Counter((x['steps']>0,x['step_verified'],x['supported']) for x in r),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('prepare','pilot','check','pairs','report'))
    p.add_argument('--name',default='pilot')
    p.add_argument('--variants',nargs='+',default=['1:407','1:413','1:420','1:427','1:430','2:420','3:420','4:420','5:420','6:420'])
    p.add_argument('--taper',action='store_true')
    p.add_argument('--feedback',type=float,default=0.)
    p.add_argument('--cutoff',type=int,default=420)
    p.add_argument('--options',type=int,nargs='+')
    args=p.parse_args()
    if args.operation=='prepare':prepare(bool(args.feedback))
    if args.operation=='pilot':pilot(args.name,[tuple(map(int,v.split(':'))) for v in args.variants])
    if args.operation=='check':verify_pilot(args.name)
    if args.operation=='pairs':pairs(args.cutoff,args.options,args.feedback,args.taper)
    if args.operation=='report':report('with_settle' if args.taper else ('with_feedback' if args.feedback else 'with'))
