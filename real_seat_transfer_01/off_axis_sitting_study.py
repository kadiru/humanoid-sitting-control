"""Compare the same catalog policies from matched lateral and angled resets."""
import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation
import torch

import furniture_skill_comparison as catalog
from fresh_sitting_skills_study import GPU, bank_digest
from run import DISK_RESERVE, P, PREPARED, read, sha, stage, write
from sitting_skills import ALIGN, Options


STUDY = P/'off_axis_sitting_01'
WORKER = P/'off_axis_sitting_worker.py'
FAMILIES = ('lateral', 'angled')
LATERAL_M = .035
YAW_DEG = 6.


def prepare():
    source = catalog.STUDY
    old = read(source/'protocol.json')
    assert len(old['roster']) == 32 and len(old['grid']) == 30
    STUDY.mkdir()
    base = STUDY/'banks'
    base.mkdir()
    bank_specs = {}
    for scale in catalog.SIZES:
        src = source/'banks'/str(scale)
        assert bank_digest(src) == old['banks'][str(scale)]
        assets = read(src/'geometry.json')
        source_rows, cases, geometry = [], [], []
        for index, asset in enumerate(assets):
            sign = (-1)**index
            for family in FAMILIES:
                row = len(cases)
                name = f"sit/{asset['name']}@{scale}/{family}"
                shift = sign*LATERAL_M if family == 'lateral' else 0.
                yaw = -sign*YAW_DEG if family == 'angled' else 0.
                cases.append(dict(row=row, case=name, asset=asset['name'], family=family,
                                  scale_pct=scale, lateral_m=shift, yaw_deg=yaw))
                geometry.append(dict(asset, case=row, surface_id=row, label=name))
                source_rows.append(index)
        out = base/str(scale)
        out.mkdir()
        with np.load(src/'bank.npz', allow_pickle=False) as z:
            arrays = {k:np.take(z[k],source_rows,axis=int(k=='q_post')) for k in z.files}
        poses = arrays['reset_root_pose']
        shift = np.asarray([c['lateral_m'] for c in cases])
        yaw = np.asarray([c['yaw_deg'] for c in cases])
        poses[:,1] += shift
        # A pure yaw at the fixed pelvis keeps the reset height and stance
        # joint coordinates from the matched standing catalog trials.
        turn = Rotation.from_rotvec(np.c_[np.zeros((len(yaw),2)),np.deg2rad(yaw)])
        poses[:,3:] = (turn*Rotation.from_quat(poses[:,3:])).as_quat()
        arrays['reference'] = np.arange(len(cases))
        np.savez_compressed(out/'bank.npz',**arrays)
        write(out/'geometry.json',geometry)
        write(out/'metadata.json',dict(read(src/'metadata.json'),cases=len(cases),ids=[c['case'] for c in cases]))
        bank_specs[str(scale)] = dict(source=str(src), hashes=bank_digest(out), cases=cases)
    settings = list(old['actor'].values())+[old['approach'][key] for key in ('checkpoint','library')]
    prior = {p:sha(p) for p in settings}
    assert len(set(prior[old['actor'][k]] for k in catalog.METHODS)) == 2
    assert all(old['inputs'][p]==h for p,h in prior.items())
    sources = [Path(__file__), WORKER, P/'furniture_skill_comparison.py', P/'sitting_skills.py',
               P/'scale_approach.py', P/'pelvis_marker_worker.py', P/'rescan_sitting.py', source/'protocol.json']
    write(STUDY/'protocol.json',dict(banks=bank_specs,grid=old['grid'],roster=old['roster'],gpu=GPU,
          approach=old['approach'],actor=old['actor'],phases=old['phases'],alignment_enabled=False,
          inputs=prior|{str(p.resolve()):sha(p) for p in sources}))
    print('Reserved 32 objects × 4 scales × 2 starts × 2 policies.',flush=True)


def rollout(method, scale):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == GPU
    protocol = read(STUDY/'protocol.json')
    assert all(sha(p)==h for p,h in protocol['inputs'].items())
    specification = protocol['banks'][str(scale)]
    source = STUDY/'banks'/str(scale)
    assert bank_digest(source)==specification['hashes']
    cases = specification['cases']
    frozen = read(catalog.REFERENCE/'evaluation.freeze.json')
    assert all(sha(p)==h for p,h in frozen['inputs'].items())
    target = STUDY/f'{method}_{scale}'
    target.mkdir()
    bank = target/'bank'
    shutil.copytree(source,bank)
    assert bank_digest(bank)==specification['hashes']
    shutil.copyfile(catalog.REFERENCE/'landing_profile.json',target/'landing_profile.json')
    write(target/'approach.json',protocol['approach'])
    write(target/'skills.json',dict(mode='evaluate',phases=protocol['phases'],choices=[[0]*4 for _ in cases],
                                    hold_rows=list(range(len(cases))),seed=73281,weights=protocol['actor'][method]))
    write(target/'reservation.json',dict(files={str(p):sha(p) for p in bank.iterdir()}))
    geometry = read(bank/'geometry.json')
    meshes = {Path(g['mesh_file']):g['mesh_sha256'] for g in geometry}
    assert all(sha(p)==h for p,h in meshes.items())
    cmd = frozen['command']
    stage(target,'evaluation',bank,PREPARED/'library.npz',worker=WORKER,extra=cmd[3:cmd.index('--mode')],
          inputs=[*map(Path,frozen['inputs']),*map(Path,protocol['inputs']),STUDY/'protocol.json',*meshes,
                  target/'skills.json',target/'approach.json',target/'landing_profile.json'],disk_reserve=DISK_RESERVE)
    catalog.score(target)


def checkpoint(path):
    state = torch.load(path,map_location='cpu',weights_only=True)
    model = Options(state['separate']).eval()
    model.load_state_dict(state['model'])
    return model


@torch.inference_mode()
def evaluate(model,features,phases):
    return model(torch.as_tensor(features,dtype=torch.float32),torch.as_tensor(phases,dtype=torch.long)).numpy()


def audit_decisions(folder, method, protocol, models, require_live):
    ev = folder/'evaluation'
    freeze = read(folder/'evaluation.freeze.json')
    assert freeze['gpu']==GPU
    path = protocol['actor'][method]
    digest = sha(path)
    assert freeze['inputs'][path]==digest
    with np.load(ev/'skill_decisions.npz',allow_pickle=False) as z:
        selected = z['enabled']
        row, phase, features, action, logp = (z[k][selected] for k in ('row','phase','features','action','logp'))
        assert not selected[z['phase']==ALIGN].any()
    expected = evaluate(models[method],features,phase)
    other = next(k for k in models if k!=method)
    alternative = evaluate(models[other],features,phase)
    np.testing.assert_array_equal(expected.argmax(1),action)
    probabilities = torch.as_tensor(expected).log_softmax(1).numpy()
    np.testing.assert_allclose(probabilities[np.arange(len(action)),action],logp,rtol=1e-5,atol=1e-6)
    if require_live:
        receipt = read(ev/'actor_receipt.json')
        assert receipt['checkpoint_sha256']==digest and receipt['calls']==len(action)
        assert receipt['separate_heads']==(method=='piecewise') and not receipt['alignment_enabled']
        with np.load(ev/'live_skill_logits.npz',allow_pickle=False) as z:
            calls, live = z['calls'],z['logits']
        np.testing.assert_array_equal(calls[:,[0,1,3]],np.c_[row,phase,action])
        assert (calls[:,2]>=0).all()
        np.testing.assert_allclose(expected,live,rtol=1e-5,atol=3e-6)
    counts = Counter(str(k) for k in phase)
    return dict(checkpoint_sha256=digest,separate_heads=models[method].separate,decisions=len(action),
                by_phase=dict(counts),counterfactual_choices_different=int((alternative.argmax(1)!=action).sum()),
                max_logit_difference=float(np.abs(expected-alternative).max()) if len(action) else None)


def audit_nominal():
    protocol = read(catalog.STUDY/'protocol.json')
    models = {k:checkpoint(p) for k,p in protocol['actor'].items()}
    assert models['piecewise'].separate and not models['unified'].separate
    replay = {m:{s:audit_decisions(catalog.STUDY/f'{m}_{s}',m,protocol,models,False)
                 for s in catalog.SIZES} for m in catalog.METHODS}
    # These are archived measured features from the earlier four-chair
    # offset comparison. Run both frozen models on exactly the same inputs.
    previous = P/'sitting_repair_transfer_24_01'/'shared'/'evaluation'/'skill_decisions.npz'
    with np.load(previous,allow_pickle=False) as z:
        mask = z['enabled'] & (z['phase']!=ALIGN)
        features,phase = z['features'][mask],z['phase'][mask]
    a,b = [evaluate(models[m],features,phase) for m in catalog.METHODS]
    disagreement = a.argmax(1)!=b.argmax(1)
    control = dict(source=str(previous),sha256=sha(previous),states=len(phase),
                   choices_different=int(disagreement.sum()),by_phase={int(p):int((disagreement & (phase==p)).sum())
                                                                      for p in np.unique(phase)})
    write(STUDY/'nominal_actor_audit.json',dict(nominal=replay,off_axis_same_input_control=control))
    print(dict(catalog_decisions={m:sum(x['decisions'] for x in replay[m].values()) for m in replay},control=control),flush=True)


def report():
    protocol = read(STUDY/'protocol.json')
    models = {k:checkpoint(p) for k,p in protocol['actor'].items()}
    summaries, receipt, rows = {}, {}, []
    for scale in catalog.SIZES:
        specification = protocol['banks'][str(scale)]
        by_method, actor = {}, {}
        for method in catalog.METHODS:
            folder = STUDY/f'{method}_{scale}'
            assert bank_digest(folder/'bank')==specification['hashes']
            actor[method] = audit_decisions(folder,method,protocol,models,True)
            by_method[method] = read(folder/'scores.json')
            assert [r['asset'] for r in by_method[method]]==[c['asset'] for c in specification['cases']]
        receipt[scale] = actor
        groups = {}
        for family in FAMILIES:
            paired = [(c,a,b) for c,a,b in zip(specification['cases'],*by_method.values(),strict=True)
                      if c['family']==family and c['asset'] in protocol['grid']]
            groups[family] = dict(starts=len(paired),piecewise=sum(a['passed'] for _,a,_ in paired),
                                  unified=sum(b['passed'] for _,_,b in paired),
                                  outcomes_differ=sum(a['passed']!=b['passed'] for _,a,b in paired),
                                  choices_differ=sum(a['skill_actions']!=b['skill_actions'] for _,a,b in paired))
        summaries[scale] = groups
        for method, data in by_method.items():
            rows.extend(dict(r,**c,method=method) for r,c in zip(data,specification['cases'],strict=True))
    assert len(rows)==len({(r['case'],r['method']) for r in rows})==512
    with (STUDY/'comparison.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(STUDY/'live_actor_audit.json',receipt)
    write(STUDY/'summary.json',dict(gpu=GPU,rollouts=len(rows),main=summaries))
    print(summaries,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','audit_nominal','run','score','report'))
    parser.add_argument('--method',choices=catalog.METHODS)
    parser.add_argument('--scale',type=int,choices=catalog.SIZES)
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare()
    elif args.operation == 'audit_nominal':
        audit_nominal()
    elif args.operation == 'run':
        rollout(args.method,args.scale)
    elif args.operation == 'score':
        catalog.score(STUDY/f'{args.method}_{args.scale}')
    else:
        report()
