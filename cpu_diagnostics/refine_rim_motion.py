"""Phase-align a legal landing and a low-slip landing using their measured first contact."""
from pathlib import Path
import argparse
import numpy as np

from compare_touchdown_ppo import verify_audit
from evaluate_rim_motion import MATCHED, PREVIOUS, bank_subset, run
from study_live_sitting import read, write


def bracket(out, previous):
    source=previous/'rollouts/screen/evaluation'
    _,rows=verify_audit(source)
    meta=read(source/'manifest.json')['metadata']
    names=meta['joint_names'][:23]
    sagittal=np.array([('hip_pitch' in n or 'ankle_pitch' in n or 'knee' in n) for n in names])
    with np.load(source/'trajectory.npz') as z:
        force=np.linalg.norm(z['physical_force_w'][:,:,0],axis=-1)/z['physical_bodyweight_N']
        actual=z['command'][:,:,:23]
        wrong=z['wrong_force']
    with np.load(previous/'screen_paths.npz') as z:
        features,indices,ids,paths=[z[k] for k in ('features','indices','ids','paths')]
    first=np.argmax(force>.01,axis=0)
    tick=np.arange(1000)
    records=[];new_paths=[];new_features=[];new_indices=[]
    def add(i,name,path):
        records.append(dict(target_index=int(indices[i]),motion=name))
        new_features.append(features[i]);new_indices.append(indices[i]);new_paths.append(path)
    for case in np.unique(indices):
        choices=np.flatnonzero(indices==case)
        clean=[i for i in choices if rows[i]['clean']]
        if clean:
            best=max(clean,key=lambda i:rows[i]['final_hold_s'])
            add(best,ids[best],paths[best])
            continue
        legal=[i for i in choices if rows[i]['stable'] and not any(rows[i]['physical_failures']) and not rows[i]['illegal_ticks']]
        best=min(legal,key=lambda i:(rows[i]['slip_ticks'],-rows[i]['final_hold_s']))
        add(best,ids[best],paths[best])
        soft=[i for i in choices if first[i]>0 and rows[i]['slip_ticks']==0]
        # Pick distinct approach shapes with little impact and small forbidden reaction.
        soft.sort(key=lambda i: (np.percentile(wrong[:,i],95), np.max(force[first[i]:first[i]+15,i])))
        selected=[]
        for i in soft:
            if all(np.max(abs(actual[first[i],i]-actual[first[j],j]))>.025 for j in selected):
                selected.append(i)
            if len(selected)==2: break
        for i in selected:
            # Keep the legal motion's hold schedule. Match the other motion's time
            # at first touch as well as at both ends of the approach and during hold.
            clock=np.interp(tick,[0,150,first[best],750,999],[0,150,first[i],750,999])
            interp=np.stack([np.interp(clock,tick,col) for col in paths[i].T],axis=-1)
            delta=interp-paths[best]
            for fraction in (.20,.4,.6,.8):
                for mask,label in ((1.,'whole'),(sagittal,'sagittal')):
                    path=(paths[best]+fraction*delta*mask).astype(np.float32)
                    add(best,f'{ids[best]} + {fraction:g} {ids[i]} {label}',path)
    assert all(np.isfinite(p).all() for p in new_paths)
    write(out/'source_motions.json',records)
    np.savez_compressed(out/'screen_paths.npz',features=new_features,paths=new_paths,indices=new_indices,
                        ids=[r['motion'] for r in records])
    bank=Path(read((MATCHED/'evaluation/manifest.json'))['args']['bank'])
    bank_subset(out/'screen_bank',bank,new_indices)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path);p.add_argument('previous',type=Path)
    args=p.parse_args()
    run(args.output.resolve(),prepare=lambda out:bracket(out,args.previous.resolve()),sources=(Path(__file__),))
