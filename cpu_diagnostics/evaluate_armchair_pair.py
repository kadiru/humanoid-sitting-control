"""Paired wooden-armchair test with attribution of the existing forbidden-force score."""
import argparse
from pathlib import Path
import sys

import numpy as np


def trace_contacts(launcher, options):
    sys.path[:0]=[str(launcher.parent),str(launcher.parent/'trainer')]
    import grip_hold_ppo
    from evaluate_chair_shapes import scene_worker
    original=grip_hold_ppo.evaluate_worker

    def run(checkpoint):
        import torch
        from environment import FullBodyEnv
        import run_supported_postures as support
        prior=FullBodyEnv.step

        def step(env,*args,**kwargs):
            tick=env.t
            result=prior(env,*args,**kwargs)
            if not kwargs.get('record'):return result
            all_sensor=env.raw.scene['all_contacts']
            if tick==0:
                env.link_audit_names=list(all_sensor.body_names)
                env.link_forces=[]
                env.filter_residuals=[]
                env.seat_channel_forces=[]
            net=all_sensor.data.net_normal_forces_w_history.torch.norm(dim=-1).amax(dim=1)
            residual=[]; seat=[]
            for key,_,_ in support.CONTACTS:
                sensor=env.raw.scene[key].data
                filtered=sensor.normal_force_matrix_w.torch.sum((1,2))
                residual.append((sensor.net_normal_forces_w.torch.sum(1)-filtered).norm(dim=-1))
            for name in getattr(env.raw.cfg,'armchair_probe_links',()):
                s=env.raw.scene['seat_probe_'+name].data
                seat.append(s.normal_force_matrix_w_history.torch.sum((2,3)).norm(dim=-1).amax(1))
            env.link_forces.append(net.cpu().numpy().copy())
            env.filter_residuals.append(torch.stack(residual,-1).cpu().numpy().copy())
            if seat:env.seat_channel_forces.append(torch.stack(seat,-1).cpu().numpy().copy())
            if tick==999:
                np.savez_compressed(env.out/'contact_attribution.npz',
                    body_names=env.link_audit_names,net_history_peak_N=env.link_forces,
                    filtered_body_names=[b for _,b,_ in support.CONTACTS],wrong_pair_N=env.filter_residuals,
                    probe_links=list(getattr(env.raw.cfg,'armchair_probe_links',())),
                    chair_history_peak_N=env.seat_channel_forces)
            return result

        # Filter separate observers to the chair rigid body; never feed them into the controller.
        make=support.make_cfg
        def cfg(args,geometry):
            c=make(args,geometry)
            links=('left_hip_pitch_link','right_hip_pitch_link','left_hip_roll_link','right_hip_roll_link',
                   'left_knee_link','right_knee_link','left_elbow_pitch_link','right_elbow_pitch_link','left_elbow_roll_link','right_elbow_roll_link','left_palm_link','right_palm_link',
                   'left_shoulder_roll_link','right_shoulder_roll_link')
            c.armchair_probe_links=links
            for name in links:
                setattr(c.scene,'seat_probe_'+name,support.base.ContactSensorCfg(
                    prim_path='{ENV_REGEX_NS}/Robot/'+name,filter_prim_paths_expr=['{ENV_REGEX_NS}/Seat'],
                    history_length=4,track_friction_forces=True))
            return c
        support.make_cfg=cfg
        FullBodyEnv.step=step
        original(checkpoint)
    grip_hold_ppo.evaluate_worker=run
    scene_worker(launcher,options)


PARENT=Path('/dev/shm/hold_clearance_02')
TARGET='hold_clearance_holdout/wooden_armchair/1'
CONTROL='rim_motion_holdout/wooden_armchair/1'


def evaluate_pair(out,phase,library=None):
    from study_live_sitting import stage, read, write, sha, PREPARED
    from evaluate_seat_plane_approach import DISK_RESERVE
    from compare_touchdown_ppo import verify_audit
    a,b=[PARENT/'rollouts'/n for n in ('fresh','previous_fresh')]
    bank=out/'pair_bank'
    refs=[(a,TARGET),(b,CONTROL)]
    if not bank.exists():
        bank.mkdir(parents=True)
        geometry=[];states={};template=None
        for j,(s,case) in enumerate(refs):
            m=read(s/'evaluation/manifest.json')
            source=Path(m['args']['bank'])
            old=read(s/'evaluation.freeze.json')
            assert all(sha(p)==old['inputs'][str(p)] for p in source.iterdir())
            meta=read(source/'metadata.json')
            i=meta['ids'].index(case)
            g=read(source/'geometry.json')[i]
            assert g['label']==case
            geometry.append(dict(g,case=j,surface_id=j))
            saved={k:v for k,v in meta.items() if k not in ('cases','ids')}
            if template is None:template=saved
            assert saved==template
            with np.load(source/'bank.npz') as z:
                for k in z:
                    if k=='reference':continue
                    val=z[k][:,i] if k=='q_post' else z[k][i]
                    states.setdefault(k,[]).append(val.copy())
        write(bank/'geometry.json',geometry)
        write(bank/'metadata.json',dict(template,cases=2,ids=[TARGET,CONTROL]))
        np.savez_compressed(bank/'bank.npz',reference=np.arange(2),
            **{k:np.stack(v,axis=1 if k=='q_post' else 0) for k,v in states.items()})
    if phase=='full':
        r=verify_audit(out/'candidate/evaluation')[1]
        assert [v['id'] for v in r]==[TARGET,CONTROL]
        assert all(v['clean'] and v['final_hold_s']>=5 for v in r) and len(r)==2
        tested=read(out/'candidate/evaluation.freeze.json')
        assert library and tested['inputs'][str(library)]==sha(library)
        assert tested['command'][tested['command'].index('--chair-contact-motion')+1]==str(library)
        bank=Path(read(a/'evaluation/manifest.json')['args']['bank'])
    study=out/phase
    study.mkdir()
    f=read(a/'evaluation.freeze.json')
    args=f['command']
    options=list(args[3:args.index('--mode')])
    assert options[0]=='--worker'
    inputs=[Path(p) for p,d in f['inputs'].items()]
    assert all(sha(p)==f['inputs'][str(p)] for p in inputs)
    if library is not None:
        options[options.index('--chair-contact-motion')+1]=str(library)
        inputs += [library,out/'construction.json',Path(__file__).with_name('armchair_approach.py')]
    inputs.extend([a/'evaluation.freeze.json',b/'evaluation.freeze.json'])
    write(study/'reservation.json',dict(files={str(p):sha(p) for p in bank.iterdir()}))
    stage(study,'evaluation',bank,PREPARED/'library.npz',worker=Path(__file__).resolve(),
          extra=options,inputs=inputs,disk_reserve=DISK_RESERVE)
    _,rows=verify_audit(study/'evaluation')
    with np.load(study/'evaluation/trajectory.npz') as d,np.load(study/'evaluation/contact_attribution.npz') as c:
        assert abs(d['command']-d['previous']).max()<=.0030003
        names=c['body_names'].tolist(); filt=c['filtered_body_names'].tolist()
        bad=[i for i,n in enumerate(names) if n not in filt]
        net=c['net_history_peak_N'][:,:,bad]
        res=c['wrong_pair_N']
        np.testing.assert_allclose(np.maximum(net.max(2),res.max(2)),d['wrong_force'],atol=.001,rtol=1e-5)
        summaries=[]
        for i,r in enumerate(rows):
            t=np.flatnonzero(d['flags'][:,i,1])
            record=dict(id=r['id'],clean=r['clean'],hold_s=r['final_hold_s'],slip_ticks=r['slip_ticks'],
                        illegal_ticks=r['illegal_ticks'],physical_failures=r['physical_failures'])
            if len(t):
                k=int(t[0]);nn=int(net[k,i].argmax());rr=int(res[k,i].argmax())
                force=float(d['wrong_force'][k,i])
                outside=(np.linalg.norm(d['physical_force_w'][k,i],axis=-1)>.01*d['weight'][k,i])&~d['physical_inside_patch'][k,i]
                name=(names[bad[nn]] if net[k,i,nn]>=res[k,i,rr] else filt[rr]) if force>10 else None
                record['outside_patch_links']=[filt[j] for j in np.flatnonzero(outside)]
                link=c['probe_links'].tolist()
                record.update(tick=k,link=name,score_force_N=float(d['wrong_force'][k,i]),
                    chair_force_N=float(c['chair_history_peak_N'][k,i,link.index(name)]) if name in link else None)
            summaries.append(record)
    write(study/'per_case.json',summaries)
    print(summaries,flush=True)


if __name__=='__main__':
    if sys.argv[1]=='--worker':
        trace_contacts(Path(sys.argv[2]),sys.argv[3:])
    else:
        p=argparse.ArgumentParser(description=__doc__)
        p.add_argument('out',type=Path);p.add_argument('phase',choices=['baseline','candidate','full'])
        p.add_argument('--library',type=Path)
        args=p.parse_args()
        evaluate_pair(args.out,args.phase,args.library)
