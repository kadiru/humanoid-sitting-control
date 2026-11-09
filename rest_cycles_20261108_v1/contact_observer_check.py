"""One frozen saved-trajectory experiment; no simulation or control deployment.

Premise: temporally aligned proprioception and camera geometry may distinguish
support loading on the already-demonstrated armchair. Whole starts are kept
separate, and the failed learned policy is an out-of-distribution challenge.
This is an empirical interval, not a force measurement or certified bound.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path('/home/kadir/humanoid-sitting/outputs/armchair_rest_ppo_20261105_v1')
FIT=['demo_rear_back_'+s for s in ['center','near','far','left','right']]
CAL=['demo_rear_back_yaw_left']
TEST=['demo_rear_back_yaw_right','learned_armchair11']
K=16


def load(name):
    p=ROOT/name/'trajectory.npz'
    with np.load(p,allow_pickle=False) as z:
        # obs[t] is pre-step; force-derived support[t] is post-step.
        # Therefore the target at obs[t] is support[t-1], NOT support[t].
        a=z['obs'];b=z['support_bw']
        assert a.shape[0]==b.shape[0]==1000 and a.shape[-1]>=333
        assert a.shape[1]==b.shape[1]
        x=a[1:,:,:333].reshape(-1,333).astype(np.float64)
        y=b[:-1].reshape(-1).astype(np.float64)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert np.all(np.diff(z['horizon'].astype(int),axis=0)>=0), 'RESET_WITHIN_TRAJECTORY'
        fail=z['failure'][:-1].reshape(-1).astype(bool)
        legal=z['legal'][:-1].reshape(-1).astype(bool)
        noslip=z['no_slip'][:-1].reshape(-1).astype(bool)
    return x,y,dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    complete_starts=int(a.shape[1]),samples=len(x),
                    failed_samples=int(fail.sum()),illegal_samples=int((~legal).sum()),slipping_samples=int((~noslip).sum()))


def geometry(x):
    f=x[:,-111:]
    return ((f[:,109]>.5)&(f[:,110]>=0)&(f[:,110]<=.21)&
            (np.linalg.norm(f[:,100:103],axis=1)>=.99)&(np.linalg.norm(f[:,100:103],axis=1)<=1.01))


def main(out):
    start=time.monotonic();out.mkdir(exist_ok=False)
    data={name:load(name) for name in FIT+CAL+TEST}
    x=np.vstack([data[n][0] for n in FIT]);y=np.concatenate([data[n][1] for n in FIT])
    center=x.mean(0);scale=np.maximum(x.std(0),.01)
    tree=cKDTree((x-center)/scale)
    def predict(q):
        d,i=tree.query((q-center)/scale,k=K,workers=1)
        return y[i].mean(1),d[:,-1]
    xc,yc,_=data[CAL[0]];pc,dc=predict(xc)
    # Fixed maximum calibration residual, using the entire calibration start.
    radius=float(np.max(abs(yc-pc)));distance_cap=float(dc.max())
    np.savez_compressed(out/'observer_model.npz',x=x,y=y,center=center,scale=scale,
                        radius=np.array(radius),distance_cap=np.array(distance_cap),k=np.array(K))
    result=dict(scope='SAVED_SENSOR_OBSERVER_VALIDATION_ONLY',sensor_only=True,
                force_truth_used_for='fit_labels_calibration_and_evaluation_only',
                excluded_inputs=['clock','task_id','case_id','initial_sensor_suffix','root_truth','force_truth_at_query'],
                alignment='obs[t] -> support_bw[t-1]; first obs omitted; no frame-shuffled split',
                fit=FIT,calibration=CAL,test=TEST,resources=dict(cpu_threads=1),
                radius_bw=radius,distance_cap=distance_cap,k=K,tests={},inputs=[data[n][2] for n in data])
    arrays={}
    for name in TEST:
        xt,yt,meta=data[name];p,d=predict(xt)
        admissible=geometry(xt)&(d<=distance_cap)
        lo=p-radius;hi=p+radius
        loaded=admissible&(lo>=.05);unloaded=admissible&(hi<.01)
        actual_loaded=yt>=.05;actual_unloaded=yt<.01
        result['tests'][name]=dict(**meta,geometry_valid=int(geometry(xt).sum()),
            in_calibration_distance=int((d<=distance_cap).sum()),
            loaded_truth=int(actual_loaded.sum()),unloaded_truth=int(actual_unloaded.sum()),
            declared_loaded=int(loaded.sum()),declared_unloaded=int(unloaded.sum()),
            false_loaded=int((loaded&~actual_loaded).sum()),false_unloaded=int((unloaded&~actual_unloaded).sum()),
            interval_misses=int((admissible&((yt<lo)|(yt>hi))).sum()),
            loaded_recall=float((loaded&actual_loaded).sum()/max(1,actual_loaded.sum())),
            unloaded_recall=float((unloaded&actual_unloaded).sum()/max(1,actual_unloaded.sum())),
            mean_abs_error_bw=float(abs(p-yt).mean()),max_abs_error_bw=float(abs(p-yt).max()))
        for key,value in dict(prediction=p,distance=d,lower=lo,upper=hi,geometry_valid=geometry(xt),
                              admissible=admissible,truth=yt,loaded=loaded,unloaded=unloaded).items():
            arrays[name+'__'+key]=value
    r=result['tests']['demo_rear_back_yaw_right']
    enough=r['loaded_recall']>=.9 and r['unloaded_recall']>=.9
    no_errors=all(t['false_loaded']==t['false_unloaded']==t['interval_misses']==0 for t in result['tests'].values())
    result['saved_data_screen_passed']=bool(enough and no_errors)
    result['deployment_admitted']=False
    result['reason']='Even a passed saved screen does not validate new furniture, unloading or stand-up states.' if enough and no_errors else 'Rejected: incomplete contact/unloaded coverage or incorrect empirical intervals on disjoint saved runs.'
    result['new_physics_steps']=result['PPO_updates']=result['complete_new_cycles']=0
    result['elapsed_seconds']=time.monotonic()-start
    np.savez_compressed(out/'observer_predictions.npz',**arrays)
    (out/'observer_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['radius_bw','saved_data_screen_passed','deployment_admitted','reason','elapsed_seconds']}),flush=True)


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
