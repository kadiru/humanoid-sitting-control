"""One-step causal pairing: sensor history[t] with contact report[t-1]."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
def main():
    manifest=json.loads((HERE/'inputs.json').read_text());freeze=json.loads((HERE/'freeze.json').read_text())
    for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    rows=[];xs=[];ys=[];groups=[];valids=[];splits=[];cache={};source_checks=[]
    for group,row in enumerate(manifest):
        path=row['path']
        if path not in cache:
            with np.load(path) as z:cache[path]={k:z[k] for k in ['sensor_history' if 'sensor_history' in z else 'obs','q_pre' if 'q_pre' in z else 'q','body','physical_force_w','physical_bodyweight_N','physical_joint_velocity']}
        d=cache[path];i=row['case']
        history=d['sensor_history'][:,i].reshape(1000,333) if 'sensor_history' in d else d['obs'][:,i,:333]
        q=d['q_pre'][:,i,:23] if 'q_pre' in d else d['q'][:-1,i,:23]
        q_error=float(np.max(abs(history[:,222:245]-q))) if len(q)==1000 else float(np.max(abs(history[1:,222:245]-q)))
        assert q_error<2e-6,(row['label'],'SENSOR_Q_ALIGNMENT',q_error)
        x=history[1:].astype(np.float32);frame=x[:,222:333]
        n=frame[:,100:103].astype(float);length=np.linalg.norm(n,axis=1)
        valid=(length>.99)&(length<1.01)&(frame[:,109]>.5)&(frame[:,110]<=.21)
        # Geometry and gravity are actor inputs; world rotation below is LABEL-ONLY.
        n=n/np.maximum(length[:,None],1e-8)
        up=-frame[:,69:72].astype(float);up/=np.maximum(np.linalg.norm(up,axis=1)[:,None],1e-8)
        e=np.where((abs(np.sum(n*up,axis=1))>.7)[:,None],np.array([1.,0.,0.]),up)
        e=e-n*np.sum(n*e,axis=1)[:,None];e/=np.maximum(np.linalg.norm(e,axis=1)[:,None],1e-8)
        b=np.cross(n,e)
        # Body0 is pelvis in the frozen robot body ordering, verified separately.
        root=Rotation.from_quat(d['body'][:-1,i,0,3:7].astype(float))
        force=root.inv().apply(d['physical_force_w'][:-1,i,:5].sum(1).astype(float))
        y=np.stack([np.sum(force*a,axis=1) for a in (n,e,b)],axis=1)/d['physical_bodyweight_N'][:-1,i,None]
        assert x.shape==(999,333) and y.shape==(999,3)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        xs.append(x);ys.append(y.astype(np.float32));groups.append(np.full(999,group));valids.append(valid)
        splits.append(np.full(999,{'train':0,'calibration':1,'test':2}[row['split']]))
        rows.append(row);source_checks.append(dict(label=row['label'],q_alignment_max_error=q_error,valid_geometry_samples=int(valid.sum()),positive_normal_samples=int((valid&(y[:,0]>.12)).sum())))
    np.savez_compressed(HERE/'dataset.npz',x=np.concatenate(xs),y=np.concatenate(ys),group=np.concatenate(groups),valid=np.concatenate(valids),split=np.concatenate(splits))
    (HERE/'dataset_audit.json').write_text(json.dumps(dict(episodes=rows,checks=source_checks,causal_offset=-1,features=333,clock_input=False,initial_signature_input=False,truth_only_for_labels=True,simulator_steps=0),indent=2))
    print('DATASET_READY',len(rows)*999,flush=True)
if __name__=='__main__':main()
