"""Summarize only independently audited, complete saved rollouts."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/home/kadir/humanoid-sitting');RUN=ROOT/'outputs/armchair_rest_ppo_20261105_v1';OUT=ROOT/'outputs/armchair_rest_policy_video_20261105_v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);protect=read(ROOT/'experiments/motion_prior_ppo_20261029_v1/protected_sitting_cases.json');report={}
    for label in ['prior_armchair11','learned_armchair11','learned_sitting72']:
        folder=RUN/label;a=read(folder/'independent_audit.json');assert a['passed'];rows=read(folder/'results.json')['episodes']
        assert sha(folder/'trajectory.npz')==a['files']['trajectory.npz']
        with np.load(folder/'trajectory.npz') as z:
            rewards=z['reward'];scores=(rewards*.999**np.arange(1000)[:,None]).sum(0)
            clean=np.array(a.get('validated_clean',a['clean']),bool);good=z['good'];norm=z['physical_normal_w'];force=z['force'];fn=np.sum(force*norm,-1)
            active=np.linalg.norm(force,axis=-1)>.01*z['weight'][...,None]
            ft=np.linalg.norm(force-fn[...,None]*norm,axis=-1);util=np.where(ft<=1e-7,0,ft/np.maximum(.6*np.maximum(fn,0),1e-7))
            checks={'contact_legality':z['legal'],'traction_margin':(~active|(util<=.90)).all(-1),'no_loaded_slip':z['no_slip'],'rest_load':z['support_bw']>=.05,'speed':z['speed']<=.1,'angular_speed':z['gyro']<=.25,'uprightness':np.abs(z['physical_trunk_tilt_rad'])<=np.deg2rad(40),'actual_joint_margin':z['physical_joint_margin_ok']}
            for i,r in enumerate(rows):
                r['final_five_second_violations']={k:int((~v[-250:,i]).sum()) for k,v in checks.items() if not v[-250:,i].all()}
                r.update(validated_clean=bool(clean[i]),reward_discounted_return=float(scores[i]),success_score_0_or_1=int(clean[i]),final_mean_knee_deg=float(np.rad2deg(a['final_mean_knee_angle_rad'][i])),final_normalized_torque_squared=a['final_normalized_torque_squared'][i],final_absolute_mechanical_power_W=a['final_absolute_mechanical_power_W'][i],final_min_rest_load_bw=float(z['support_bw'][-250:,i].min()),final_max_loaded_slip_m_s=float(z['loaded_slip'][-250:,i].max()),final_max_active_friction_util=float(np.where(active[-250:,i],util[-250:,i],0).max()),first_failure_s=float((np.flatnonzero(z['failure'][:,i])[0]+1)*.02) if z['failure'][:,i].any() else None)
        report[label]=dict(clean=int(clean.sum()),total=len(rows),cases=rows,trajectory_sha256=a['files']['trajectory.npz'])
    rows=report['learned_sitting72']['cases'];repair=[r['id'] for r in rows if r['id'] in protect['failed'] and r['validated_clean']];regress=[r['id'] for r in rows if r['id'] in protect['protected'] and not r['validated_clean']]
    before=report['prior_armchair11']['cases'];after=report['learned_armchair11']['cases']
    report['comparison']=dict(original_sitting_repairs=repair,original_sitting_regressions=regress,armchair_repairs=[b['id'] for a,b in zip(before,after) if b['validated_clean'] and not a['validated_clean']],armchair_regressions=[b['id'] for a,b in zip(before,after) if a['validated_clean'] and not b['validated_clean']],same_success_set_effort_delta=[dict(id=b['id'],relative_change=b['final_normalized_torque_squared']/a['final_normalized_torque_squared']-1) for a,b in zip(before,after) if a['validated_clean'] and b['validated_clean']],automatically_promoted=False)
    # Correct a logging-only ambiguity: the generic initial-selection record
    # reports rest-library indices even during the explicitly sitting branch.
    import torch
    ck=torch.load(RUN/'training_65011/final.pt',map_location='cpu',weights_only=False)
    with np.load(RUN/'learned_sitting72/trajectory.npz') as z:
        sig=z['obs'][0,:,336:];lib=ck['sitting'];distance=np.mean(((sig[:,None]-lib['signature'].numpy()[None])/lib['signature_scale'].numpy())**2,axis=-1);choice=distance.argmin(-1)
        expected=lib['desired'].numpy()[choice].transpose(1,0,2)
        error=float(np.max(np.abs(z['desired']-expected)));assert error==0.,('SITTING_BRANCH_DRIFT',error)
    (OUT/'sitting_selection_log_correction.json').write_text(json.dumps(dict(scope='Saved-observation reconstruction; logging correction only; no simulation rerun',reason='The original generic debug record returned rest-library choice indices even when the sitting branch supplied the action.',actual_sitting_library_indices=choice.tolist(),actual_desired_commands_equal_frozen_sitting_prior=True,max_difference_rad=error),indent=2)+'\n')
    report['learning']=read(RUN/'training_65011/final_audit.json');report['limitations']=['Simulation only; one armchair mesh and narrow pose range.','New residual disabled for sitting; retention is architectural preservation, not PPO relearning.','Original16 leaning motions preserved but not physically reevaluated in this run.','Pelvis-rest prototype failed before intended support and was excluded. No validated straight-leg resting.','Torque squared and mechanical power are proxies; no electrical energy measurement.','Reserved4 armchair starts are separate from the untouched original24 holdout.']
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='cases'} for k,v in report.items() if k in ['comparison','learning','prior_armchair11','learned_armchair11','learned_sitting72']},indent=2))
if __name__=='__main__':main()
