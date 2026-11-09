"""Audit applied controller bounds and kinematic-estimator error against labels."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);a=p.parse_args();d=a.folder
    read=lambda n:json.loads((d/n).read_text())
    cfg=read('landing_config.json');result=read('results.json');names=result['joint_names']
    ids=[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    other=[i for i in range(37) if i not in ids];finger=[i for i,n in enumerate(names) if any('_'+s+'_' in n for s in ['zero','one','two','three','four','five','six'])]
    assert len(finger)==14
    with np.load(d/'trace.npz') as data:trace={k:data[k] for k in data.files}
    with np.load(d/'startup.npz') as data:start={k:data[k] for k in data.files}
    assert np.array_equal(trace['command'][:,:,finger],np.broadcast_to(start['command'][None,:,finger],trace['command'][:,:,finger].shape))
    records=read('landing_controller_audit.json');correction=np.asarray([r['correction'] for r in records])
    assert np.isfinite(correction).all() and np.max(abs(correction))<=cfg['joint_cap']+1.e-12
    assert np.count_nonzero(correction[:,:,other])==0
    origin=np.asarray(result['origins']);pelvis=result['body_names'].index('pelvis');errors=[];translations=[]
    for record in records:
        t=record['tick']*.02;at=int(np.argmin(abs(trace['t']-t)))
        if abs(float(trace['t'][at])-t)>1.e-6:continue
        for r in record['states']:
            translations.append(r['delta']);actual=trace['body'][at,r['case'],pelvis,:3]-origin[r['case']]
            errors.append(np.asarray(r['position'])-actual)
    if translations:assert np.max(abs(np.asarray(translations)))<=cfg['translation_cap']+1.e-12
    load_path=d/'load_feedback_audit.json';max_load=0.;max_step=0.
    if load_path.exists():
        previous={}
        for record in read('load_feedback_audit.json'):
            for r in record['states']:
                i=r['case'];v=r['applied_fraction'];assert np.isfinite(v) and 0<=v<=1.
                step=abs(v-previous.get(i,0.));assert step<=cfg['load_fraction_step']+1.e-12
                max_step=max(max_step,step);previous[i]=v
                c=np.asarray(r['correction']);assert c.shape==(6,) and np.isfinite(c).all()
                assert np.max(abs(c))<=cfg['load_command_cap']+1.e-12;max_load=max(max_load,float(np.max(abs(c))))
    out=dict(passed=True,uses_truth_for_this_audit_only=True,fixed_fingers_exact=True,
        max_damping_joint_correction_rad=float(np.max(abs(correction))),
        max_load_joint_correction_rad=max_load,max_load_fraction_step=max_step,
        estimated_position_error_max_abs_m=np.max(abs(np.asarray(errors)),axis=0).tolist() if errors else None,
        estimate_sample_count=len(errors),no_physical_success_implied=True)
    dest=d/'sensor_control_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2));print(json.dumps(out))

if __name__=='__main__':main()
