"""Use achieved clean landings and pre-impact load plateaus to finish a housing repair."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from armchair_approach import register, transfer
from prepare_sitting_repairs import BodyModel, MODEL
from repair_right_hip_contacts import BASE, HIP, bank, collect, contact_report, save_screen, sha, source_actors, targets
from seat_contact_arrest import construct
from study_live_sitting import read


def refine(out, prior):
    screen = prior/'screen'
    rows = contact_report(screen)
    info = read(prior/'construction.json')['probes']
    with np.load(prior/'paths.npz') as z:
        original = z['paths']
    gs = read(bank(BASE)/'geometry.json')
    meta = read(bank(BASE)/'metadata.json')
    limits = [np.array(meta[k]) for k in ('lower','upper')]
    model = BodyModel(MODEL)
    trials = collect(screen, *source_actors())
    records, motions = [], []
    def add(i,j,path,method,**data):
        assert path.shape == (1000,23) and np.isfinite(path).all()
        if any(r['target']==i and np.max(abs(p-path))<1e-5 for p,r in zip(motions,records)):
            return
        records.append(dict(target=i, source=str(screen), source_row=j, donor=info[j]['donor'],
                            method=method, **data))
        motions.append(path.copy())
    clean = [j for j,r in enumerate(rows) if r['clean'] and r['hip_ticks']==0]
    f = screen/'evaluation'
    with np.load(f/'trajectory.npz') as z,np.load(f/'edge_audit.npz') as edges:
        good = z['good']
        flags = z['flags']
        load = z['support_bw']
        points = edges['pelvis_contact_world_local']
    for i in targets():
        verified = [j for j in clean if info[j]['target']==i]
        if verified:
            for j in sorted(verified,key=lambda j:-rows[j]['hold_s'])[:2]:
                add(i,j,original[j],'verified approach')
            continue
        # Capture a reached command inside an uninterrupted, legal load interval.
        # Static positions correspond to increasing support loads before the impact.
        for j,r in enumerate(rows):
            if info[j]['target'] != i or r['slips']:
                continue
            hit = np.flatnonzero(flags[:,j,1])
            usable = np.flatnonzero(good[:hit[0] if len(hit) else len(good),j])
            if len(usable)<5:
                continue
            group = max(np.split(usable,np.flatnonzero(np.diff(usable)!=1)+1),key=len)
            if len(group)<5:
                continue
            trial=trials[j]
            for frac in (.20,.50,.80):
                t=int(group[round(frac*(len(group)-1))])
                path=original[j].copy()
                path[t+1:]=trial['command'][t]
                path[750:]-=trial['residual']
                add(i,j,path,'hold reached support pose',tick=t,measured_support_bw=float(load[t,j]))
            for method,path,witness in construct(screen,j,trial,meta,gs[i],limits,model):
                add(i,j,path,method,witness=witness)
        # Transfer the actually achieved contact from the most similar repaired surfaces.
        # Rank by the difference in seat height, normal and usable front inset at first load.
        donors=[]
        for j in clean:
            g=gs[info[j]['target']]
            ok=np.flatnonzero((load[:,j]>.01)&np.isfinite(points[:,j]).all(axis=1))
            if not len(ok):
                continue
            touch=int(ok[0])
            point=points[touch,j]
            rv,shift=transfer(g,gs[i],point)
            inset=[]
            for surface in (g,gs[i]):
                R=Rotation.from_quat(surface['rotation_xyzw']).as_matrix()
                inset.append(float(surface['depth']/2-((point-surface['top_center'])@R)[0]))
            cost=np.linalg.norm(shift)+.2*np.linalg.norm(rv)+.6*abs(inset[0]-inset[1])
            donors.append((cost,j,touch,point,rv,shift,inset))
        chosen=[]
        for cost,j,touch,point,rv,shift,inset in sorted(donors,key=lambda item:item[0]):
            if any(abs(original[k][350:820:10]-original[j][350:820:10]).max()<.012 for k in chosen):
                continue
            chosen.append(j)
            add(i,j,original[j],'transferred clean approach',insets_m=inset)
            if len(chosen)<=3:
                witness=dict(id=rows[j]['id'],contact=touch,pivot=point.tolist(),
                             rotation_vector=rv.tolist(),shift=shift.tolist(),original_hold_s=rows[j]['hold_s'])
                fit=register(screen,j,witness,model)
                if fit:
                    add(i,j,fit[0],'registered clean approach',insets_m=inset,fit=fit[1])
            if len(chosen)>=6:
                break
    inputs={str(path):sha(path) for path in (prior/'paths.npz',prior/'construction.json',screen/'evaluation/independent_audit.json',Path(__file__))}
    save_screen(out,records,motions,inputs)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('prior',type=Path)
    a=p.parse_args()
    refine(a.out,a.prior)
