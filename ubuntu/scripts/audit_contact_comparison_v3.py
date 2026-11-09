"""Independent reward reconstruction, per-step CPU replay and reset identity."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
from feedback_posture_ppo import FeedbackPPO
from audit_contact_reward_v3 import oracle,geometry_oracle
from contact_reward_v3 import LandingLabels
from audit_contact_reset_v2 import exact_arrays
from audit_checkpoint_updates_v2 import audit as weight_audit
from contact_reward_v1 import reward_to_go
torch.set_num_threads(1)

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def reset(root,old):
    root,old=Path(root),Path(old);new=root/'identity_zero';ref=old/'calibrate'
    for name in ['startup.npz','traction_startup.npz','trace.npz','applied_offsets.npz','sensor_inputs.npz']:
        with np.load(ref/name) as a,np.load(new/name) as b:exact_arrays(dict(a),dict(b))
    a,b=read(ref/'results.json')['episodes'],read(new/'results.json')['episodes']
    assert len(a)==len(b)==48
    for x,y in zip(a,b):
        for k in ['case','profile','success','steps','duration_s','failures','slip_ticks','illegal_ticks','traction_bad_ticks','final_hold_s']:
            assert x[k]==y[k],('RESET_RESULT',k)
    assert read(new/'feedback_contract.json')['local_updates']==0
    write(root/'reset_identity_audit.json',dict(passed=True,reference=str(ref),episodes=48,exact_saved_arrays=True,tolerance_changed=False))

def audit(folder):
    p=Path(folder);batches=load(p/'decision_batches.pt');initial=load(p/'initial.pt');rows=read(p/'results.json')
    with np.load(p/'trace.npz') as z:trace={'body':z['body']}
    with np.load(p/'every_tick.npz') as z:ticks=dict(z)
    with np.load(p/'reward_geometry.npz') as z:geom=dict(z)
    cal=read(read(p/'manifest.json')['feedback_args']['feedback_calibration'])
    matched=cal['bank_sha256']==read(p/'manifest.json')['bank_hashes']
    margin=cal['margin_ticks'] if matched else [0]*24
    gs=read(p/'geometry.json');label=LandingLabels(rows['body_names'],[gs[r['geometry_index']] for r in rows['episodes'][:24]],rows['origins'])
    train=read(p/'feedback_contract.json')['mode']=='train';reports=[]
    offsets=np.load(p/'applied_offsets.npz')['offsets']
    assert offsets.shape==(200*len(batches),3,24,3)
    assert np.abs(offsets[:,0]).max()<=.00400001 and np.abs(offsets[:,2]).max()<=.01200001
    residual=offsets[:,0].reshape(len(batches),200,24,3)
    assert np.abs(np.diff(residual,axis=1)).max()<=.00040001
    for it,b in enumerate(batches):
        r={k:v[it*1025:it*1025+1000] for k,v in ticks.items()};g={k:v[it] for k,v in geom.items()};erows=rows['episodes'][it*24:it*24+24]
        assert len(erows)==24 and all(x['steps']==1000 for x in erows)
        assert np.max(np.abs(r['t']-np.arange(1,1001)*.02))<1e-9
        expected,control=oracle(r,g,erows,margin,b['decision_steps'])
        error=float(np.max(np.abs(expected-b['revised_reward'].numpy())))
        assert error<2e-7,error
        original=reward_to_go(b['torque_integrals'],b['baseline_integrals'],b['terminal_penalty'],b['decision_steps'],b['physical_cost'])
        assert torch.allclose(original,b['original_reward'],atol=2e-5,rtol=1e-6)
        assert torch.equal(b['reward'],b[b['reward_arm']+'_reward'])
        for k in [0,60,119,199]:
            pose=trace['body'][it*205+k];ref=geometry_oracle(label,pose)
            saved=np.stack([g['pelvis_target_distance_m'][k],g['housing_vertex_clearance_m'][k]],axis=1)
            assert np.max(np.abs(ref-saved))<1e-10
        record=dict(update=initial['updates']+it+1,reward_max_abs_error=error)
        if train:
            # Each replay starts from the actual preceding GPU checkpoint.
            # This tests every update without accumulating cross-device drift.
            before=p/('initial.pt' if it==0 else f'model_{initial["updates"]+it:03d}.pt')
            after=load(p/f'model_{initial["updates"]+it+1:03d}.pt')
            model=FeedbackPPO(initial['input_mean'],initial['input_scale'],63111,'cpu');model.load(before)
            stats=model.update(b['obs'].flatten(0,1),b['actions'].flatten(0,1),b['logp'].flatten(),b['old_mean'].flatten(0,1),b['reward'].flatten())
            errors={name:max(float((v-after[name+'_state_dict'][key]).abs().max()) for key,v in getattr(model,name).state_dict().items()) for name in ['actor','critic']}
            assert max(errors.values())<2e-4,errors
            actual=read(p/'learning_audit.json')[it]['optimizer']
            assert stats['actor_steps']==actual['actor_steps'] and stats['critic_steps']==actual['critic_steps']
            assert model.updates==after['updates']==initial['updates']+it+1
            record['cpu_replay_parameter_error']=errors
        reports.append(record)
    if train:weight_audit(p,f'model_{initial["updates"]+len(batches):03d}.pt')
    report=dict(passed=True,per_update=reports,training=train,full_episodes=len(rows['episodes']),physical_repair_claimed=False)
    write(p/'independent_audit.json',report);print(json.dumps(dict(passed=True,folder=str(p),updates=len(batches),training=train)),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='reset':reset(sys.argv[2],sys.argv[3])
    else:audit(sys.argv[1])
