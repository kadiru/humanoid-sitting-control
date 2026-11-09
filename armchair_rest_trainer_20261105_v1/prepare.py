"""Prepare immutable motion priors and explicit standing-start banks; CPU only."""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from common import ROOT,OBS,read,write,sha
from policy import Learner,command,teacher_targets
DEMO=ROOT/'outputs/armchair_rest_ppo_20261105_v1'
OLD=ROOT/'outputs/full_body_contact_imitation_v4/dataset'
RESERVED=[dict(id='reserved_near_left',x=-.0075,y=.02,yaw_deg=1.5),dict(id='reserved_near_right',x=-.0075,y=-.02,yaw_deg=-1.5),dict(id='reserved_far_left',x=.0075,y=.02,yaw_deg=-1.5),dict(id='reserved_far_right',x=.0075,y=-.02,yaw_deg=1.5)]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    status=read(DEMO/'demo_selection.json');assert status['complete'];out=a.output;out.mkdir(exist_ok=False)
    variants=read(ROOT/'experiments/armchair_rest_ppo_20261105_v1/protocol.json')['demonstrations']['variants']
    chosen=[]
    for var in variants:
        candidates=[r for r in status['results'] if r.get('variant',{}).get('id')==var['id'] and r['admitted']]
        if candidates:chosen.append(min(candidates,key=lambda r:r['final_normalized_torque_squared']))
    assert chosen,'NO_CLEAN_DEMONSTRATIONS'
    meta=read(OLD/'leaning/metadata.json');lo=torch.tensor(meta['lower']);hi=torch.tensor(meta['upper']);initials=[];targets=[];new=[];source=[];reconstruction=[]
    with np.load(OLD/'leaning/bank.npz') as z:legacy=dict(z)
    keep=np.flatnonzero(legacy['admitted']);assert len(keep)==16
    obs=legacy['obs'][:,keep];sig=obs[0,:,222:333];initials.extend(sig)
    desired=teacher_targets(torch.from_numpy(legacy['previous'][:,keep,:23]),torch.from_numpy(legacy['command'][:,keep,:23]),lo,hi).numpy()
    targets.extend(desired.transpose(1,0,2));new.extend([0.]*len(keep));source.extend(['preserved_leaning/'+str(i) for i in keep])
    full=[];records=[]
    for row in chosen:
        folder=DEMO/row['label'];audit=read(folder/'independent_audit.json');assert audit['admitted']
        assert sha(folder/'trajectory.npz')==audit['source_hashes']['trajectory.npz']
        with np.load(folder/'trajectory.npz') as z:d=dict(z)
        full.append(d);obs=d['obs'][:,0];initials.append(obs[0,336:])
        desired=teacher_targets(torch.from_numpy(d['previous'][:,0,:23]),torch.from_numpy(d['command'][:,0,:23]),lo,hi)
        running=torch.from_numpy(d['previous'][0,0,:23].copy());errors=[]
        for t in range(1000):
            running=command(running,desired[t],lo,hi);errors.append(float((running-torch.from_numpy(d['command'][t,0,:23])).abs().max()))
        assert max(errors)<=2e-7,('COMMAND_RECURRENCE',row['label'],max(errors));reconstruction.append(max(errors))
        targets.append(desired.numpy());new.append(1.);source.append(row['label']);records.append(obs)
    initials=np.array(initials,dtype=np.float32);library=dict(signature=initials,signature_scale=np.maximum(initials.std(0),.05),desired=np.array(targets,dtype=np.float32),new_skill=np.array(new,np.float32))
    original_path=ROOT/'outputs/motion_prior_ppo_20261029_v1/prepared/initial_64111.pt'
    original=torch.load(original_path,map_location='cpu',weights_only=False);assert all(torch.count_nonzero(original['actor'][k])==0 for k in ['4.weight','4.bias'])
    x=np.concatenate(records).astype(np.float32);agent=Learner(x.mean(0),x.std(0),65011);agent.bind_library(library);agent.bind_sitting(original['library']);agent.bind_demonstrations(x)
    with torch.no_grad():
        starts=torch.from_numpy(np.array([r[0] for r in records]));pred=agent.choose(starts,False)[0].numpy()
        expected=np.array(targets)[16:,0];assert np.max(abs(pred-expected))<=2e-7,'AMBIGUOUS_NEW_INITIAL_SENSOR'
    agent.save(out/'initial.pt');np.savez_compressed(out/'demonstrations.npz',obs=x)
    for name,vs in [('train',variants),('evaluation',variants+RESERVED)]:
        folder=out/name;folder.mkdir();n=len(vs);bank={}
        for k in ['reset_q','reset_qd','reset_command','reset_root_pose','reset_root_velocity']:bank[k]=np.repeat(legacy[k][:1],n,axis=0)
        qpost=[];phases=[]
        for i,v in enumerate(vs):
            bank['reset_root_pose'][i,:2]+=np.array([v['x'],v['y']])
            R=Rotation.from_euler('z',v['yaw_deg'],degrees=True);bank['reset_root_pose'][i,3:7]=(R*Rotation.from_quat(bank['reset_root_pose'][i,3:7])).as_quat()
            for sl in [slice(0,3),slice(3,6)]:bank['reset_root_velocity'][i,sl]=R.apply(bank['reset_root_velocity'][i,sl])
            distances=[((v['x']-r['variant']['x'])/.015)**2+((v['y']-r['variant']['y'])/.04)**2+((v['yaw_deg']-r['variant']['yaw_deg'])/3)**2 for r in chosen]
            ix=int(np.argmin(distances));d=full[ix];qpost.append(d['q'][:,0]);load=d['support_bw'][:,0]>.05;good=d['good'][:,0]
            phases.append([.02*(1+np.flatnonzero(load)[0]),.02*(1+np.flatnonzero(good)[0])])
        bank.update(q_post=np.stack(qpost,axis=1),reference=np.arange(n),phase_times=np.array(phases,np.float32),admitted=np.ones(n,bool))
        m=copy.deepcopy(meta);m.update(cases=n,ids=[v['id'] for v in vs],positive_demonstrations=len(chosen),variants=vs,reference_is_reward_only=True,reset_slots_used=[0],fresh_holdout_used=False)
        np.savez_compressed(folder/'bank.npz',**bank);write(folder/'metadata.json',m)
    write(out/'preparation.json',dict(passed=True,selected=chosen,library_sources=source,new_demonstrations=len(chosen),preserved_sitting_demonstrations=64,preserved_leaning_demonstrations=16,recurrence_max_error_rad=max(reconstruction),initial_source=str(original_path),initial_source_sha256=sha(original_path),training_variants=variants,reserved_variants=RESERVED,prior_is_sensor_selected=True,task_intent_selects_sitting_or_resting=True,claim='Construction/reconstruction only; physical policy performance unvalidated.'))
if __name__=='__main__':main()
