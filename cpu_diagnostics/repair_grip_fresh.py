"""Construct and verify sitting repairs using audited shifted-seat trajectories."""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from align_sitting_touchdown import align, counterbalance
from compare_touchdown_ppo import BASELINE, BANKS, summaries, verify_audit
from evaluate_grip_fresh import RETAINED_REFERENCE, write_results
from live_sitting_edge import CAMERA_HANDOFF_CHECK_TICK
from prepare_sitting_repairs import BodyModel, encode, leg_fk, smoothstep
from repair_live_touchdown import CHECKPOINT, HERE, MODEL, PRIOR, ROOT, applied_path, planned_path, read, sha, write
from repair_touchdown_neighborhoods import geometry_adjusted
from study_live_sitting import stage
from touchdown_feedback import REPAIR_PREFIXES
from validate_camera_touchdown import fallback_priors

from common import OBS
from policy import Learner


SHIFTED = ROOT/'outputs/touchdown_grip_fresh_20261116_v2'
PREVIOUS = ROOT/'outputs/touchdown_grip_ppo_comparison_20261115_v1'
FAILURES = ('02','03','04','11','12','16','21','24')
GRIP = {'04','11','12','16'}
FEATURE_SCALE = np.array([.006,.005,.8])
DONOR_SCALE = np.array([.022,.064,.016,3.4])
FOLLOWUP = ROOT/'outputs/touchdown_grip_repaired_fresh_20261116_v1'
REPAIRED = ROOT/'outputs/touchdown_grip_shift_repair_20261116_full'
RETAINED_FRESH = RETAINED_REFERENCE.parent/'grip_retained_332_fresh_20261116_v1'


class RepairBank:
    def __init__(self,followup,retained=False):
        self.retained = retained
        self.followup = followup or retained
        if retained:
            self.source,self.previous = RETAINED_FRESH,RETAINED_REFERENCE
            self.failures = ('01','02','03','04','08','11','12','16','20','23')
            self.total,self.layer_sizes = 356,(8,10)
        elif followup:
            self.source,self.previous = FOLLOWUP,REPAIRED
            self.failures = ('01','02','03','04','08','11','12','16','23','24')
            self.total,self.layer_sizes = 332,(8,)
        else:
            self.source,self.previous = SHIFTED,PREVIOUS
            self.failures = FAILURES
            self.total,self.layer_sizes = 308,()
        self.grip = {'01','03','04','08','11','12','16','23'} if self.followup else GRIP
        self.prefix = REPAIR_PREFIXES[len(self.layer_sizes)]
        self.protected_count = self.total-len(self.failures)

        self.freeze = read(self.source/'evaluation.freeze.json')
        command = self.freeze['command']
        self.options = command[3:command.index('--mode')]
        assert self.options[0]=='evaluate'
        self.checkpoint = Path(self.options[1])
        self.routes = Path(self.options[self.options.index('--feedback-routes')+1])
        if self.followup:
            prior = read(self.previous/'status.json')
            assert prior == dict(state='complete',total=self.total-24,clean=self.total-24,regressions=[])
            assert self.routes == self.previous/'routes.npz'
            for filename,digest in self.freeze['inputs'].items():
                assert sha(filename)==digest,filename

    def retained_runs(self):
        if self.retained:
            summary = read(self.previous/'retention.json')
            assert summary['clean']==summary['total']==self.total-24 and not summary['regressions']
            runs = {}
            for label,score in summary['banks'].items():
                assert score['clean']==score['total'] and not score['regressions']
                if label=='shifted':
                    folder,name = self.previous/'evaluation','second_shifted'
                else:
                    folder,name = self.previous/label,label
                manifest = read(folder/'manifest.json')
                bank = Path(manifest['args']['bank'])
                assert sha(bank/'bank.npz')==manifest['bank_sha256']
                runs[name] = (folder,bank)
            return runs
        runs = {label:(self.previous/label,bank) for label,(_,bank) in previous_runs().items()}
        if self.followup:
            runs['prior_shifted'] = (self.previous/'evaluation',SHIFTED/'bank')
        return runs


def previous_runs():
    saved = read(BASELINE/'ppo_gate.json')['evidence']
    geometries = read(BASELINE/'paired_gate.json')['banks']
    return {label:(Path(saved[label]),Path(geometries[label])) for label in BANKS}


def utilization(data):
    force,normal = (data['physical_'+k][-250:,:,:5] for k in ('force_w','normal_w'))
    fn = (force*normal).sum(-1)
    tangential = np.linalg.norm(force-fn[...,None]*normal,axis=-1)
    loaded = np.linalg.norm(force,axis=-1)>.01*data['physical_bodyweight_N'][-250:,:,None]
    capacity = np.maximum(data['physical_friction'][-250:,:,:5]*np.maximum(fn,0),1e-7)
    return np.where(loaded,tangential/capacity,0.).max(axis=(0,2))


def load_paths(folder,indices,actor_checkpoint=None):
    audit,rows = verify_audit(folder)
    manifest = read(folder/'manifest.json')
    bank = Path(manifest['args']['bank'])
    geometry = read(bank/'geometry.json')
    result = read(folder/'results.json')
    pelvis = result['body_names'].index('pelvis')
    ankle = [result['body_names'].index(f'{side}_ankle_roll_link') for side in ('left','right')]
    with np.load(folder/'trajectory.npz',allow_pickle=False) as data:
        prior = fallback_priors(data,indices)
        if actor_checkpoint is not None:
            learner = Learner(np.zeros(OBS),np.ones(OBS),65011)
            learner.load(CHECKPOINT)
            trained = copy.deepcopy(learner.actor)
            trained.load_state_dict(torch.load(actor_checkpoint,map_location='cpu',weights_only=False)['actor'])
            obs = torch.from_numpy(data['obs'][750:,indices].copy()).flatten(0,1)
            with torch.no_grad():
                change = ((trained(learner.normalize(obs))-learner.actor(learner.normalize(obs)))*
                          learner.residual_mask(obs)).numpy().reshape(250,len(indices),23)
            tick = np.arange(250)[:,None]
            phase = np.minimum(tick/50,1)
            active = np.array(read(folder/'grip_handoff.json')['active_ticks'])[indices]
            amount = phase**2*(3-2*phase)*(tick<=active[None,:])
            prior[:,750:] -= (change*amount[...,None]).transpose(1,0,2)
        pose,q,load,old,slip,valid = (data[k] for k in ('body','q','support_bw','previous','loaded_slip','no_slip'))
    output = {}
    for col,i in enumerate(indices):
        item = geometry[i]
        origin = np.asarray(result['origins'][i])
        root = pose[:,i,pelvis].copy()
        root[:,:3] -= origin
        foot = pose[0,i,ankle,:3].mean(0)-origin
        seat = np.asarray(item['top_center'])
        forward = Rotation.from_quat(item['rotation_xyzw']).as_matrix()[:,0]
        normal = np.asarray(item['normal'])
        features = [(seat-foot)@forward+item['depth']/2,(seat-foot)@normal/normal[2],
                    item['pitch_deg'],seat[1]-foot[1]]
        output[rows[i]['id']] = dict(index=i,row=rows[i],geometry=item,feature=np.array(features),
             prior=prior[col],previous=old[0,i],motion=dict(body=root,q=q[:,i],load=load[:,i]),
             slip=slip[:,i],bad=np.flatnonzero(~valid[:,i]))
    return output,{str(folder/k):v for k,v in audit['files'].items()}


def sort_donors(targets,references,grip):
    catalog = []
    for folder,bank in references.values():
        _,rows = verify_audit(folder)
        with np.load(folder/'trajectory.npz',allow_pickle=False) as data:
            ratio = utilization(data)
            slip = data['loaded_slip'].max(0)
        for i,(seat,row) in enumerate(zip(read(bank/'geometry.json'),rows,strict=True)):
            if row['clean'] and row['final_hold_s']>=5.3:
                feature = seat['top_center']+[seat['pitch_deg']]
                catalog.append((folder,i,row['id'],feature,float(ratio[i]),float(slip[i])))
    result = {}
    for name,target in targets.items():
        seat = target['geometry']
        geometry = seat['top_center']+[seat['pitch_deg']]
        def cost(donor):
            gap = np.square((np.asarray(donor[3])-geometry)/DONOR_SCALE).sum()
            grip_weight = 1.8 if name[-2:] in grip else .45
            return gap+grip_weight*(donor[4]/.9)**3+.15*(donor[5]/.02)**2
        result[name] = sorted(catalog,key=cost)[:12]
    return result


def prepare(out,study):
    original = study.source/'evaluation'
    _,rows = verify_audit(original)
    targets = [i for i,r in enumerate(rows) if not r['clean']]
    assert tuple(rows[i]['id'][-2:] for i in targets)==study.failures
    cases,inputs = load_paths(original,targets,study.checkpoint)
    references = study.retained_runs() if study.followup else previous_runs()
    if study.followup:
        references['new_passes'] = (original,study.source/'bank')
    ranking = sort_donors(cases,references,study.grip)
    donors = {}
    for folder,_ in references.values():
        indices = sorted({i for matches in ranking.values() for p,i,*_ in matches if p==folder})
        if indices:
            sample,files = load_paths(folder,indices,study.checkpoint if study.followup else None)
            donors.update({(folder,v['index']):v for v in sample.values()})
            inputs.update(files)
    source = study.source/'bank'
    metadata = read(source/'metadata.json')
    limits = [np.asarray(metadata[k]) for k in ('lower','upper')]
    model = BodyModel(MODEL)
    fk,legs = leg_fk(model,metadata['joint_names'])
    with np.load(original/'camera_touchdown_selection.npz',allow_pickle=False) as z:
        camera = z['features']
    assert np.isfinite(camera[targets]).all()
    shots,paths,poses,rejected = [],[],[],[]

    for name,target in cases.items():
        distinct = []
        for folder,index,*_ in ranking[name]:
            candidate = donors[folder,index]
            if all(np.max(np.abs(candidate['prior'][:750]-old['prior'][:750]))>.012 for old in distinct):
                distinct.append(candidate)
            if len(distinct)==4:
                break

        def add(method,donor,construct):
            try:
                path,checks = construct()
            except (AssertionError,ValueError) as error:
                rejected.append(dict(case=name,method=method,donor=donor,reason=str(error)))
                return
            new = np.asarray(path,np.float32)
            assert new.shape==(1000,23) and np.isfinite(new).all()
            if any(r['case']==name and np.array_equal(p,new) for r,p in zip(shots,paths,strict=True)):
                return
            shots.append(dict(id=f'probe/{len(shots)+1:02d}',case=name,source_index=target['index'],
                              method=method,donor=donor,checks=checks))
            paths.append(new)
            poses.append(target['geometry'])

        needs_donor = name[-2:] in study.grip
        if not needs_donor:
            for method,params in [('retime',dict(speed=.72)),('settle',dict(lead=62,delay=13.))]:
                add(method,name,lambda:planned_path(target,target,dict(method=method,**params),model,fk,legs,limits))
            add('counterbalance',name,lambda:counterbalance(target['prior'],target['previous'],
                    target['motion']['load'],metadata['joint_names'],limits,.12,.14))
        donor_count = (3 if study.followup else 4) if needs_donor else 2
        for candidate in distinct[:donor_count]:
            key = candidate['row']['id']
            add('replay',key,lambda:(candidate['prior'],{}))
            add('retarget',key,lambda:geometry_adjusted(candidate,target,model,fk,legs,limits,-.002,.005))
        if study.followup and needs_donor:
            candidate = distinct[0]
            add('clearance',candidate['row']['id'],
                lambda:geometry_adjusted(candidate,target,model,fk,legs,limits,-.006,.024))
    assert 0<len(shots)<=72 and {r['case'] for r in shots}==set(cases)
    out.mkdir()
    build_probe(out,source,metadata,shots,poses,paths,camera,study)
    write(out/'candidates.json',dict(rows=shots,rejected=rejected,source_traces=inputs))


def pack_routes(study,shots,paths,camera,initial,limits):
    with np.load(study.routes,allow_pickle=False) as data:
        library = {k:data[k] for k in data.files if not k.startswith(study.prefix)}
    repair_start = len(library['features'])
    for prefix,count in zip(REPAIR_PREFIXES[:len(study.layer_sizes)],study.layer_sizes,strict=True):
        assert prefix+'rows' not in library and len(library[prefix+'features'])==count
        repair_start += count
    handoff = []
    for shot,path in zip(shots,paths,strict=True):
        i = shot['source_index']
        handoff.append(applied_path(path[:CAMERA_HANDOFF_CHECK_TICK+1],initial[i],*limits)[-1])
    fields = dict(features=camera[[r['source_index'] for r in shots]],scale=FEATURE_SCALE,
                  radius=np.ones(len(shots)),paths=np.asarray(paths),handoff=np.asarray(handoff))
    library.update({study.prefix+key:value for key,value in fields.items()})
    library['ids'] = np.concatenate((library['ids'][:repair_start],
                                    np.array([r['case']+' '+r['method'] for r in shots])))
    return library


def build_probe(out,source,metadata,shots,poses,paths,camera,study):
    indices = [r['source_index'] for r in shots]
    source_bank = np.load(source/'bank.npz',allow_pickle=False)
    with source_bank as data:
        arrays = {k:data[k][indices] for k in data.files if k.startswith('reset_')}
        initial = data['reset_command']
    library = pack_routes(study,shots,paths,camera,initial,
                          [np.asarray(metadata[k]) for k in ('lower','upper')])
    bank = out/'bank'
    bank.mkdir()
    arrays.update(reference=np.arange(len(shots)),admitted=np.ones(len(shots),bool),
                  q_post=np.broadcast_to(arrays['reset_q'],(1000,len(shots),37)),
                  phase_times=np.full((len(shots),2),15.))
    np.savez_compressed(bank/'bank.npz',**arrays)
    write(bank/'geometry.json',[dict(g,case=i,surface_id=i) for i,g in enumerate(poses)])
    write(bank/'metadata.json',dict(metadata,cases=len(shots),ids=[r['id'] for r in shots]))
    write(out/'reservation.json',dict(files={str(p):sha(p) for p in bank.iterdir()}))
    library[study.prefix+'rows'] = np.arange(len(shots))
    np.savez_compressed(out/'routes.npz',**library)


def refine(out,study,probe,transfer=None):
    winners,inputs = select_winners([probe],study)
    unresolved = set(study.failures)-{name[-2:] for name in winners}
    assert study.retained and unresolved=={'02','11'},unresolved
    if transfer is not None:
        corrected,more_inputs = select_winners([transfer],study)
        assert not corrected
        inputs.update(more_inputs)
    source = study.source/'bank'
    rows = read(study.source/'evaluation/results.json')['episodes']
    indices = [i for i,r in enumerate(rows) if r['id'][-2:] in unresolved]
    targets,files = load_paths(study.source/'evaluation',indices,study.checkpoint)
    inputs.update(files)
    shallow,steeper = targets.values()
    assert (shallow['row']['id'][-2:],steeper['row']['id'][-2:])==('02','11')
    meta = read(source/'metadata.json')
    limits = tuple(np.asarray(meta[k]) for k in ('lower','upper'))
    model = BodyModel(MODEL)
    fk,legs = leg_fk(model,meta['joint_names'])
    shots,paths,poses,rejected = [],[],[],[]

    def add(target,method,donor,construct):
        name = target['row']['id']
        try:
            path,checks = construct()
        except (AssertionError,ValueError) as error:
            rejected.append(dict(case=name,method=method,donor=donor,reason=str(error)))
            return
        path = np.asarray(path,np.float32)
        assert path.shape==(1000,23) and np.isfinite(path).all()
        shots.append(dict(id=f'probe/{len(shots)+1:02d}',case=name,
                          source_index=target['index'],method=method,donor=donor,checks=checks))
        paths.append(path)
        poses.append(target['geometry'])

    # Candidate 11 had no slip. Its first loaded contact was outside the
    # scored patch at the front edge. Set the clean donor deeper before loading.
    scores = read(probe/'screen_results.json')
    near = scores[10]
    assert near['id']=='probe/11' and near['slip_ticks']==0 and near['illegal_ticks']==1
    donors,files = load_paths(study.previous/'evaluation',[1],study.checkpoint)
    inputs.update(files)
    donor = donors[near['donor']]
    for inset in ((-.0025,-.003,-.0035,-.004) if transfer else (-.005,-.008,-.011,-.014)):
        for pitch in ((0.,.005) if transfer else (.005,.018)):
            add(shallow,'inset',near['donor'],
                lambda:geometry_adjusted(donor,shallow,model,fk,legs,limits,inset,pitch))

    # Both paths remained legal but slid briefly during load transfer.
    # Bring the pelvis to its later settled position before taking the load.
    near_ids = ('probe/36','probe/37','probe/11') if transfer else ('probe/36','probe/37')
    near_indices = [i for i,r in enumerate(scores) if r['id'] in near_ids]
    screened,files = load_paths(probe/'probe',near_indices,study.checkpoint)
    inputs.update(files)
    with np.load(probe/'routes.npz',allow_pickle=False) as z:
        prior_paths = z[study.prefix+'paths']
    for i in near_indices:
        receipt = scores[i]
        assert receipt['routed']
        case = screened[receipt['id']]
        path = prior_paths[i]
        initial = case['previous']
        command = applied_path(path,initial,*limits)
        pose,q,load = (case['motion'][k] for k in ('body','q','load'))
        if receipt['id']=='probe/11':
            with np.load(probe/'probe/trajectory.npz',allow_pickle=False) as z:
                illegal = np.flatnonzero(~z['legal'][:,i])
            assert len(illegal)==1 and receipt['slip_ticks']==0
            for lift in (.0008,.0014,.002):
                add(shallow,'delay_edge_load',receipt['id'],lambda:align(
                    path,initial,pose,q,command,load,illegal,shallow['geometry'],
                    model,fk,legs,limits,lift,0.,0.))
            continue
        assert receipt['illegal_ticks']==0 and receipt['hold_s']>=7
        gains = ((.22,0.,0.),(.34,0.,0.),(.48,0.,0.),(.34,-.001,.0008)) if transfer else (
                 (.65,0.,0.),(.9,0.,0.),(.8,.002,.002),(.8,-.002,.002))
        for gain,offset,lift in gains:
            add(steeper,'align',receipt['id'],
                lambda:align(path,initial,pose,q,command,load,case['bad'],steeper['geometry'],
                             model,fk,legs,limits,lift,offset,gain))
        if not transfer:
            for speed in (.65,.82):
                add(steeper,'settle_slip',receipt['id'],lambda:planned_path(
                    dict(case,prior=path),case,dict(method='retime',speed=speed),model,fk,legs,limits))
    if transfer:
        # Compare genuinely different landings from the retained banks.
        # Rank by geometry and measured traction, excluding screened donors.
        references = study.retained_runs()
        ranking = sort_donors(targets,references,study.grip)
        for name,target in targets.items():
            tried = {r['donor'] for r in scores if r['case']==name}
            alternatives = [r for r in ranking[name] if r[2] not in tried][:7]
            for folder,index,key,*_ in alternatives:
                sample,files = load_paths(folder,[index],study.checkpoint)
                inputs.update(files)
                donor = sample[key]
                add(target,'alternate_replay',key,lambda:(donor['prior'],{}))
                add(target,'alternate_retarget',key,lambda:geometry_adjusted(
                    donor,target,model,fk,legs,limits,-.002,.005))
    assert {r['case'][-2:] for r in shots}==unresolved
    camera_file = study.source/'evaluation/camera_touchdown_selection.npz'
    with np.load(camera_file,allow_pickle=False) as z:
        camera = z['features']
    inputs[str(camera_file)] = sha(camera_file)
    out.mkdir()
    build_probe(out,source,meta,shots,poses,paths,camera,study)
    write(out/'candidates.json',dict(rows=shots,rejected=rejected,source_traces=inputs))


def dwell_before_load(path,initial,limits,load_tick,lead,duration):
    clock = np.arange(len(path),dtype=float)
    center = load_tick-lead
    delay = duration*smoothstep((clock-center+7)/14)
    delay *= 1-smoothstep((clock-load_tick-28)/(745-load_tick-28))
    schedule = clock+delay
    assert np.diff(schedule).min()>0 and delay[745]==0
    applied = applied_path(path,initial,*limits)
    desired = np.stack([np.interp(clock,schedule,k) for k in applied.T],axis=1)
    latent,decoded = encode(desired,initial,*limits)
    error = float(np.abs(decoded-desired).max())
    assert error<=.004,error
    start = center-7
    result = path.copy()
    result[start:746] = latent[start:746]
    return result,dict(load_tick=load_tick,center_tick=center,delay_ticks=duration,encoding_error_rad=error)


def refine_dwell(out,study,probe,later):
    winners,inputs = select_winners([probe,later],study)
    unresolved = set(study.failures)-{name[-2:] for name in winners}
    assert study.retained and unresolved=={'11'},unresolved
    source = study.source/'bank'
    original = study.source/'evaluation'
    scores = read(probe/'screen_results.json')
    indices = [i for i,r in enumerate(scores) if r['id'] in ('probe/33','probe/36','probe/37')]
    cases,files = load_paths(probe/'probe',indices,study.checkpoint)
    inputs.update(files)
    meta = read(source/'metadata.json')
    limits = tuple(np.asarray(meta[k]) for k in ('lower','upper'))
    target = 'retained_grip_fresh/11'
    target_index = meta['ids'].index(target)
    geometry = read(source/'geometry.json')[target_index]
    with np.load(probe/'routes.npz',allow_pickle=False) as z:
        priors = z[study.prefix+'paths']
    shots,paths,rejected = [],[],[]

    def add(donor,method,construct):
        try:
            path,checks = construct()
        except (AssertionError,ValueError) as error:
            rejected.append(dict(case=target,method=method,donor=donor,reason=str(error)))
            return
        path = np.asarray(path,np.float32)
        assert path.shape==(1000,23) and np.isfinite(path).all()
        shots.append(dict(id=f'probe/{len(shots)+1:02d}',case=target,source_index=target_index,
                          method=method,donor=donor,checks=checks))
        paths.append(path)

    for i in indices:
        donor = scores[i]['id']
        case = cases[donor]
        initial,load = case['previous'],case['motion']['load']
        contact = int(np.flatnonzero(load>.05)[0])
        assert scores[i]['routed'] and scores[i]['illegal_ticks']==0
        for lead in (7,14,21):
            for duration in (12,22,30):
                add(donor,'dwell',lambda:dwell_before_load(priors[i],initial,limits,contact,lead,duration))
        for shoulder,elbow in ((.055,.065),(.10,.12),(.16,.15),(.19,.16)):
            add(donor,'counterbalance',lambda:counterbalance(
                priors[i],initial,load,meta['joint_names'],limits,shoulder,elbow))
    camera_file = original/'camera_touchdown_selection.npz'
    with np.load(camera_file,allow_pickle=False) as z:
        camera = z['features']
    inputs[str(camera_file)] = sha(camera_file)
    assert shots
    out.mkdir()
    build_probe(out,source,meta,shots,[geometry]*len(shots),paths,camera,study)
    write(out/'candidates.json',dict(rows=shots,rejected=rejected,source_traces=inputs))


def run_stage(out,label,bank,routes,protocol,study):
    options = study.options.copy()
    options[options.index('--feedback-routes')+1] = str(routes)
    return stage(out,label,bank,PRIOR,worker=HERE/'grip_hold_ppo.py',extra=options,protocol=protocol,
          inputs=[*HERE.glob('*.py'),routes,
                  *[Path(p) for p in study.freeze['inputs']]])


def screen(out,study,refinement=None,transfer=None,dwell=None):
    if dwell is not None:
        refine_dwell(out,study,refinement,dwell)
    elif refinement is None:
        prepare(out,study)
    else:
        refine(out,study,refinement,transfer)
    protocol = out/'candidates.json'
    rows = read(protocol)['rows']
    run_stage(out,'probe',out/'bank',out/'routes.npz',protocol,study)
    folder = out/'probe'
    _,episodes = verify_audit(folder)
    with np.load(folder/'trajectory.npz',allow_pickle=False) as z, \
         np.load(folder/'camera_touchdown_selection.npz',allow_pickle=False) as routes:
        ratio = utilization(z)
        slips = z['loaded_slip'].max(0)
        selected = [str(routes['ids'][i]) if i>=0 else '' for i in routes['selected']]
    summary = []
    for i,(row,result) in enumerate(zip(rows,episodes,strict=True)):
        assert row['id']==result['id']
        routed = selected[i]==row['case']+' '+row['method']
        summary.append(dict(row,clean=result['clean'] and routed,routed=routed,hold_s=result['final_hold_s'],
             slip_ticks=result['slip_ticks'],illegal_ticks=result['illegal_ticks'],
             peak_slip=float(slips[i]),peak_hold_utilization=float(ratio[i])))
    write(out/'screen_results.json',summary)


def select_winners(probes,study):
    winners = {}
    inputs = {}
    for probe in probes:
        receipt = probe/'screen_results.json'
        routes = probe/'routes.npz'
        frozen = probe/'probe.freeze.json'
        assert read(frozen)['inputs'][str(routes)]==sha(routes)
        audit,episodes = verify_audit(probe/'probe')
        inputs.update({str(probe/'probe'/p):digest for p,digest in audit['files'].items()})
        inputs.update({str(p):sha(p) for p in (receipt,routes,frozen)})
        with np.load(routes,allow_pickle=False) as data:
            paths = data[study.prefix+'paths']
        for i,(row,episode) in enumerate(zip(read(receipt),episodes,strict=True)):
            assert row['id']==episode['id']
            if not row['clean']:
                continue
            assert row['routed'] and episode['clean']
            cost = row['peak_slip']/.02 + row['peak_hold_utilization']/.9
            old = winners.get(row['case'])
            if old is None or cost<old[0]:
                winners[row['case']] = cost,row,paths[i],probe
    return winners,inputs


def certify(out,probes,study):
    winners,inputs = select_winners(probes,study)
    assert {name[-2:] for name in winners}==set(study.failures),sorted(winners)
    source = study.source/'bank'
    original = study.source/'evaluation'
    _,rows = verify_audit(original)
    protected = []
    references = study.retained_runs()
    for folder in [*(folder for folder,_ in references.values()),original]:
        audit,episodes = verify_audit(folder)
        passed = [i for i,row in enumerate(episodes) if row['clean']]
        with np.load(folder/'trajectory.npz',allow_pickle=False) as data:
            protected.extend(data['obs'][0,passed,336:])
        inputs.update({str(folder/p):v for p,v in audit['files'].items()})
    assert len(protected)==study.protected_count
    with np.load(original/'trajectory.npz',allow_pickle=False) as z:
        target = z['obs'][0,[i for i,row in enumerate(rows) if not row['clean']],336:]
    with np.load(BASELINE/'protected.npz',allow_pickle=False) as z:
        scale = z['scale']
    separation = np.square((target[:,None]-np.asarray(protected))/scale).mean(-1).min(1)
    assert (separation>5e-7).all(),separation.tolist()
    with np.load(original/'camera_touchdown_selection.npz',allow_pickle=False) as z, \
         np.load(source/'bank.npz',allow_pickle=False) as bank:
        meta = read(source/'metadata.json')
        winners = [winners[name] for name in sorted(winners)]
        shots = [dict(w[1],probe=str(w[3])) for w in winners]
        library = pack_routes(study,shots,[w[2] for w in winners],z['features'],bank['reset_command'],
                              [np.asarray(meta[k]) for k in ('lower','upper')])
    out.mkdir()
    routes = out/'routes.npz'
    library[study.prefix+'radius'][:] = .25
    library[study.prefix+'protected'] = np.asarray(protected)
    np.savez_compressed(routes,**library)
    banks = [source,*(bank for _,bank in references.values())]
    write(out/'reservation.json',dict(files={str(p):sha(p) for bank in banks for p in bank.iterdir()}))
    inputs.update({str(p):sha(p) for p in (routes,Path(__file__),HERE/'touchdown_feedback.py',study.routes)})
    protocol = out/'repair_protocol.json'
    write(protocol,dict(inputs=inputs,repairs=shots,protected_passes=study.protected_count,
                         min_protected_signature_distance=separation.tolist(),source_bank=str(source),
                         action_limits_unchanged=True,scoring_unchanged=True))
    display = out/'bank'
    display.mkdir()
    (display/'geometry.json').write_bytes((source/'geometry.json').read_bytes())
    score = run_stage(out,'evaluation',source,routes,protocol,study)
    write_results(out,read(source/'geometry.json'),score)


def rerun(out,study):
    protocol = out/'repair_protocol.json'
    for name,digest in read(protocol)['inputs'].items():
        assert sha(name)==digest,name
    score = read(out/'comparison.json')['score']
    restored = summaries(study.source/'evaluation',out/'evaluation')
    data = {'shifted':dict(score,repairs=[row['id'] for row in restored if row['ppo_clean'] and not row['touchdown_clean']],
                          regressions=[row['id'] for row in restored if row['touchdown_clean'] and not row['ppo_clean']])}
    retained = [r for r in restored if r['touchdown_clean']]
    for phase in ('approach','hold'):
        key = phase+'_max_command_delta_rad'
        data['shifted']['protected_'+key] = max(r[key] for r in retained)
    if study.followup:
        assert score['clean']==score['total']==24
        assert not data['shifted']['regressions']
    result = out/'retention.json'
    write(result,dict(banks=data))
    for name,(before,bank) in study.retained_runs().items():
        score = run_stage(out,name,bank,out/'routes.npz',protocol,study)
        check = summaries(before,out/name)
        data[name] = dict(score,regressions=[r['id'] for r in check if r['touchdown_clean'] and not r['ppo_clean']],
                         command_max_delta_rad=max(max(r['approach_max_command_delta_rad'],r['hold_max_command_delta_rad'])
                                                   for r in check))
        write(result,dict(banks=data))
    totals = dict(total=sum(r['total'] for r in data.values()),clean=sum(r['clean'] for r in data.values()),
                  regressions=[i for bank in data.values() for i in bank['regressions']],banks=data)
    assert totals['total']==study.total
    write(result,totals)
    write(out/'status.json',dict(state='complete',**{k:totals[k] for k in ('total','clean','regressions')}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--mode',choices=('screen','certify','rerun'),default='screen')
    parser.add_argument('--probe',type=Path,nargs='+')
    parser.add_argument('--refine-from',type=Path)
    parser.add_argument('--transfer-from',type=Path)
    parser.add_argument('--dwell-from',type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--followup',action='store_true')
    group.add_argument('--retained',action='store_true')
    parsed = parser.parse_args()
    torch.set_num_threads(1)
    study = RepairBank(parsed.followup,parsed.retained)
    if parsed.mode=='screen':
        screen(parsed.output.resolve(),study,parsed.refine_from,parsed.transfer_from,parsed.dwell_from)
    elif parsed.mode=='certify':
        certify(parsed.output.resolve(),[p.resolve() for p in parsed.probe],study)
    else:
        rerun(parsed.output.resolve(),study)
