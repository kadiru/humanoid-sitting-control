"""Replay the first touchdown warning from pre-action sensors on independent placement banks."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluate_chair_contact_motion import signatures
from hold_clearance import StanceHull, signals
from prepare_sitting_repairs import BodyModel, MODEL
from study_live_sitting import read, sha, write


# Entire banks are kept together. A pose repeated across splits is kept in the later split.
CALIBRATION = {'coverage_matched', 'known_fresh', 'known_retain_development', 'known_retain_prior_shifted'}
VALIDATION = {'prior_matched', 'known_latest', 'known_retain_test', 'known_retain_positioned'}
TICKS = np.arange(250, 1000, 5)
LEG = np.array([0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20])
LOW, HIGH = 5, 35  # lead in 50 Hz control steps


def features(obs, scan, hull):
    # The command and torques here are the values the robot already knows before applying this action.
    ticks = np.r_[TICKS[0]-20, TICKS[0]-15, TICKS[0]-10, TICKS[0]-5, TICKS]
    frame = obs[ticks, 222:333]
    q = frame[:, LEG]
    vel = 10.*frame[:, 23+LEG]
    torque = frame[:, 46+LEG]
    err = frame[:, 75+LEG]-q
    body = np.column_stack((frame[:, 69:75], q, vel, torque, err))
    if np.isfinite(scan[:4]).all():
        geom = signals(hull, frame, scan, ticks)
    else:
        geom = np.full((len(ticks), 4), np.nan)
    current = np.column_stack((geom, body))
    # Each feature ends at or before the decision tick. No centered smoothing.
    rate = (current[4:]-current[:-4])/.4
    if np.isfinite(scan[:4]).all():
        gap = geom[4:,:3]
        closing = (geom[:-2,:3]-geom[2:,:3])[2:]/.2
        predicted_gap = gap-.35*np.maximum(closing, 0.)
    else:
        predicted_gap = np.full((len(TICKS), 3), np.nan)
    return np.column_stack((TICKS*.02, np.tile(scan[:4], (len(TICKS),1)), current[4:],
                            rate, predicted_gap)).astype(np.float32)


def planned_features(obs, scan, hull, path, delay, lower, upper):
    """Forecast from today's command, frozen plan and measured posture, holding tracking error constant."""
    x = np.full((len(TICKS), 15), np.nan, dtype=np.float32)
    # After tick 750 the live actor supplies part of the future command.
    valid = TICKS+30<=750
    ticks = TICKS[valid]
    if path is None:
        return x, None
    frame = obs[ticks,222:333].copy()
    origin = frame[:,75:98].copy()
    command = origin.copy()
    first_command = None
    direct = np.isfinite(scan[:4]).all()
    if direct:
        _, vertex, axes, center, _ = hull.sample(frame,scan)
        base = np.einsum('nij,nj->ni',axes,vertex-center)
    for offset in range(30):
        latent = path[ticks+offset-delay[valid]]
        command += np.clip(.1*(np.clip(latent,lower+.03,upper-.03)-command),-.003,.003)
        if offset==0:
            first_command = command.copy()
        steps = offset+1
        if direct and steps in (10,20,30):
            predicted = frame.copy()
            predicted[:,:23] += command-origin
            gaps,_,future_axes,future_center,normal = hull.sample(predicted,scan)
            future = np.einsum('nij,nj->ni',future_axes,vertex-future_center)
            vel = (future-base)/(.02*steps)
            approach = (vel*normal).sum(axis=1)
            tangential = np.linalg.norm(vel-approach[:,None]*normal,axis=1)
            start = 5*(steps//10-1)
            x[valid,start:start+5]=np.column_stack((gaps,tangential,approach))
    return x, first_command


def collect(root, preview=False):
    banks = read(root/'summary.json')['known']
    assert sum(v['total'] for v in banks.values()) == 596 and CALIBRATION|VALIDATION <= banks.keys()
    traces = []
    paths = None
    if preview:
        with np.load(root/'confirmed.npz') as lib:
            paths=lib['paths']
        expected_digest=sha(root/'confirmed.npz')
    for label in [*banks, 'fresh']:
        stage = root/'rollouts'/label
        p = stage/'evaluation'
        manifest = read(p/'manifest.json')
        assert manifest['action_limits_unchanged'] and manifest['physical_criteria_unchanged']
        meta = manifest['metadata']
        placements = read(Path(manifest['args']['bank'])/'geometry.json')
        audit = read(p/'independent_audit.json')
        assert audit['passed'] and audit['strict_predicates_exact']
        assert sha(p/'trajectory.npz') == audit['files']['trajectory.npz']
        hull = StanceHull(BodyModel(MODEL), meta['joint_names'])
        scans = signatures(stage)
        episodes = read(p/'results.json')['episodes']
        with np.load(p/'trajectory.npz') as z:
            obs, slips, forbidden = z['obs'], ~z['no_slip'], z['flags'][:,:,1]
        with np.load(p/'hold_guard.npz') as z:
            guard = z['trace']
        assert len(episodes)==len(placements)==obs.shape[1] and obs.shape[0]==1000
        if preview:
            freeze=read(stage/'evaluation.freeze.json')
            assert freeze['inputs'][str(root/'confirmed.npz')]==expected_digest
            with np.load(p/'chair_contact_motion.npz') as z:
                selected=z['selected']
            with np.load(p/'trajectory.npz') as z:
                applied=z['command']
                alive=z['alive']
        for i,(ep,pose,scan) in enumerate(zip(episodes,placements,scans,strict=True)):
            assert ep['id']==meta['ids'][i] and ep['steps']==1000
            events = np.flatnonzero(slips[:,i]|forbidden[:,i])
            onset = int(events[0]) if len(events) else -1
            kind = 'none' if onset<0 else '+'.join(k for k,v in [('slip',slips[onset,i]),('forbidden',forbidden[onset,i])] if v)
            orig = guard[guard[:,1]==i]
            fired = orig[(orig[:,5:7]!=0).any(1)]
            # Optimistic incumbent comparison: make its residual alarm available on all logged original paths.
            earlier = int(fired[0,0]) if len(fired) else -1
            x = features(obs[:,i], np.asarray(scan), hull)
            if preview:
                delay=np.zeros(len(TICKS),int)
                if len(orig):
                    ix=np.searchsorted(orig[:,0],TICKS,side='right')-1
                    use=ix>=0
                    delay[use]=orig[ix[use],7].astype(int)
                path=paths[selected[i]] if selected[i]>=0 else None
                extra,one_step=planned_features(obs[:,i],np.asarray(scan),hull,path,delay,
                                               np.asarray(meta['lower']),np.asarray(meta['upper']))
                # Audit only: the feature calculation never sees these executed actions.
                if one_step is not None:
                    t=TICKS[TICKS+30<=750]
                    keep=alive[t-1,i]
                    np.testing.assert_allclose(one_step[keep],applied[t[keep],i,:23],atol=3e-7,rtol=0)
                x=np.column_stack((x,extra))
            posekey = tuple(np.round(pose['top_center']+pose['rotation_xyzw'],8))
            split = 'challenge' if label=='fresh' else 'audit' if label in VALIDATION else 'calibration' if label in CALIBRATION else 'fit'
            traces.append(dict(bank=label,id=ep['id'],clean=ep['clean'],onset=onset,kind=kind,
                               incumbent=earlier,split=split,pose=posekey,x=x))
    order = {'fit':0,'calibration':1,'audit':2,'challenge':3}
    latest = {}
    for r in traces:
        latest[r['pose']] = max(latest.get(r['pose'],-1),order[r['split']])
    for r in traces:
        if order[r['split']]<latest[r['pose']]:
            r['split']='overlapping_pose'
    return traces


def fit(rows, timer=False):
    xx, yy, ww = [], [], []
    for r in rows:
        onset = r['onset']
        mask = (TICKS<=onset-LOW) if onset>=0 else np.ones(len(TICKS),bool)
        if not mask.any():continue
        x = r['x'][mask]
        y = (onset>=0)&(TICKS[mask]>=onset-HIGH)
        w = np.ones(len(x))/len(x)
        # Each positive episode has equal positive weight regardless of event time.
        if y.any():
            w[y]=1./y.sum()
        xx.append(x[:,:1] if timer else x)
        yy.append(y)
        ww.append(w)
    model = HistGradientBoostingClassifier(max_iter=70,max_leaf_nodes=9,min_samples_leaf=45,
            learning_rate=.08,l2_regularization=2.,early_stopping=False,random_state=9182)
    w=np.concatenate(ww)
    return model.fit(np.vstack(xx), np.concatenate(yy), sample_weight=w/w.mean())


def first_alarm(scores,threshold):
    yes = scores>threshold
    idx = np.flatnonzero(yes[1:]&yes[:-1])
    return int(TICKS[idx[0]+1]) if len(idx) else -1


def alert_class(tick,onset):
    if tick<0:return 'silent'
    if onset<0:return 'alarm_without_event'
    lead=onset-tick
    if lead>HIGH:return 'early'
    if lead<LOW:return 'late'
    return 'timely'


def run(root,out,preview=False):
    out.mkdir(parents=True,exist_ok=False)
    records=collect(root,preview=preview)
    splits=Counter(r['split'] for r in records)
    train=[r for r in records if r['split']=='fit']
    cal=[i for i,r in enumerate(records) if r['split']=='calibration' and r['clean']]
    assert cal and train and splits['challenge']==24
    rows=[{k:v for k,v in r.items() if k not in ('x','pose')} for r in records]
    scores={}
    for name in ['timer','sensor']:
        model=fit(train,timer=name=='timer')
        # Set the boundary using only training-independent clean calibration episodes.
        # Strictly exceeding the largest consecutive-pair score gives zero calibration interruptions.
        x=np.concatenate([r['x'] for r in records])
        proba=model.predict_proba(x[:,:1] if name=='timer' else x)[:,1].reshape(len(rows),-1)
        cut=float(np.minimum(proba[cal,1:],proba[cal,:-1]).max())
        joblib.dump(dict(model=model,threshold=cut,ticks=TICKS),out/(name+'.joblib'))
        scores[name]=proba
        for row,p in zip(rows,proba,strict=True):row[name]=first_alarm(p,cut)
    np.savez_compressed(out/'scores.npz',ticks=TICKS,**scores)
    for row in rows:
        for policy in ['incumbent','timer','sensor']:
            row[policy+'_result']=alert_class(row[policy],row['onset'])
    with (out/'paired_warnings.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    totals={}
    for split in ('fit','calibration','audit','challenge','overlapping_pose'):
        group=[r for r in rows if r['split']==split]
        if not group:continue
        totals[split]={}
        for policy in ['incumbent','timer','sensor']:
            outcomes=Counter(r[policy+'_result'] for r in group if r['onset']>=0)
            totals[split][policy]=dict(cases=len(group),events=sum(outcomes.values()),
                clean=sum(r['clean'] for r in group),interrupted_clean=sum(r['clean'] and r[policy]>=0 for r in group),
                before_event=dict(outcomes),missed_cleanliness_without_event=sum(not r['clean'] and r['onset']<0 for r in group))
    write(out/'report.json',dict(source_sha256=sha(Path(__file__).resolve()),experiment=str(root),planned_motion_preview=preview,split_counts=dict(splits),
            model_sha256={k:sha(out/(k+'.joblib')) for k in scores},results=totals))
    print(json.dumps(totals,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--preview',action='store_true',help='Include an open-loop forecast of the selected chair motion.')
    args=p.parse_args()
    run(args.archive.resolve(),args.output.resolve(),args.preview)
