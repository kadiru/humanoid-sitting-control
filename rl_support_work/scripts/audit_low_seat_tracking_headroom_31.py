"""Audit new bounded rate-headroom experiment; old joint17 failures remain unchanged."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from humanoid_sitting.shared_frame import BodyModel
from audit_low_seat_contact_probe_v2 import summarize,clean
from validate_low_positive_feedback_v3 import validate
from low_seat_joint_path_controller_17 import plan
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');ROOT=Path('/home/kadir/humanoid-sitting')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,required=True);args=ap.parse_args();folder=args.folder;p=folder/'candidate'
    read=lambda x:json.loads(Path(x).read_text());torch.set_num_threads(2)
    v=validate(p);a=read(p/'rolling_plan_audit.json');selected=sorted(x['case'] for x in a if 'joint_path_17' in x)
    assert selected==[0,1,6,7,10,11]
    m=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json');inertia_path=Path(read(p/'bridge_contract.json')['args']['bridge_inertial']);inertia=read(inertia_path)
    assert hashlib.sha256(inertia_path.read_bytes()).hexdigest()==read(p/'bridge_sources.json')[str(inertia_path)]
    cal=read(D.parent/'contact_effort_repair_01/plane_calibration.json');cfg=read(p/'landing_config.json');paths=np.load(p/'last_rolling_paths.npz');replay=[]
    trace=np.load(p/'trace.npz');at=int(np.argmin(abs(trace['t']-8.)))
    for row in a:
        if 'joint_path_17' not in row:continue
        assert row['tick']==400
        data=row['joint_path_17']['runtime_inputs'];i=row['case']
        for key in ['q','geometry','command']:assert np.array_equal(np.array(data[key]),trace[key][at,i]),(i,key,'actual planner input')
        assert data['reference'][0]==float(trace['estimated_root_x'][at,i])
        cs,qs,info=plan(m,**{k:np.asarray(data[k],dtype=np.float64 if k=='reference' else np.float32) for k in ['q','gravity','geometry','reference','command']},cal=cal,inertial=inertia,config=cfg)
        i=row['case'];ce=float(abs(paths['command'][i]-cs).max());qe=float(abs(paths['q'][i]-qs).max())
        assert ce<2e-7 and qe<2e-7
        replay.append(dict(case=i,command_max_error=ce,q_max_error=qe,screen=info['joint_path_17']['screen']))
    starts={}
    for n in ['startup.npz','traction_startup.npz','bridge_startup.npz']:
        b=np.load(folder/'baseline'/n);c=np.load(p/n);starts[n]={k:bool(np.array_equal(b[k],c[k])) for k in b.files};assert all(starts[n].values())
    b=np.load(folder/'baseline/trace.npz');c=np.load(p/'trace.npz');best=np.load(D/'normal_landing3s_06/candidate/trace.npz')
    assert np.array_equal(b['t'],c['t']);prefix={};unchanged={};other=[i for i in range(24) if i not in selected]
    for k in ['body','q','command','reference','geometry','torque']:
        mask=b['t']<=8.;prefix[k]=bool(np.array_equal(b[k][mask],c[k][mask]));assert prefix[k]
        unchanged[k]=dict(exact=bool(np.array_equal(best[k][:,other],c[k][:,other])),max_error=float(abs(best[k][:,other]-c[k][:,other]).max()))
    # Model timing changes must not silently alter any unselected episode.
    unselected_exact=all(x['exact'] for x in unchanged.values())
    # Record this failed contract explicitly; never turn it into an admitted run.
    assert np.isfinite(c['command']).all();step=float(abs(np.diff(c['command'],axis=0)).max());assert step<=.020001
    br=read(folder/'baseline/results.json')['episodes'];cr=read(p/'results.json')['episodes'];bestrows=read(D/'normal_landing3s_06/candidate/results.json')['episodes']
    common=[i for i,(x,y) in enumerate(zip(bestrows,cr)) if clean(x) and clean(y)];effort={}
    for window in ['metrics_whole_trial','metrics_last10s','metrics_last5s']:
        effort[window]={}
        for key in ['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2']:
            x=np.array([bestrows[i][window][key] for i in common]);y=np.array([cr[i][window][key] for i in common]);effort[window][key]=dict(n=len(x),baseline=float(x.mean()),candidate=float(y.mean()),percent_change=float((y.mean()/x.mean()-1)*100))
    ticks=np.load(p/'command_headroom_ticks_31.npz');assert len(ticks['t'])==1000
    selection=ticks['selected'];expected=np.zeros_like(selection)
    expected[:,selected]=((ticks['t'][:,selected]>=8.)&(ticks['t'][:,selected]<15.))
    assert np.array_equal(selection,expected) and ticks['active'].all()
    expectedcap=np.where(expected,.004,.003);assert np.allclose(ticks['cap'],expectedcap,atol=2e-10,rtol=0.)
    delta=abs(ticks['after']-ticks['before']);assert np.all(delta<=expectedcap[:,:,None]+3e-7)
    finger=[i for i,n in enumerate(m.names) if any(x in n for x in ['thumb','index','middle'])];assert len(finger)==14
    assert np.isfinite(ticks['after']).all() and np.array_equal(ticks['after'][:,:,finger],ticks['before'][:,:,finger])
    assert v['actor_steps']==0 and v['critic_steps']==0
    out=dict(audit_completed=True,passed=unselected_exact,software_audit_passed=True,exact_preservation_gate_passed=unselected_exact,full20_exactness_is_reported_not_relaxed=True,command_rate_change_explicit=True,actual_command_tick_audit=dict(ticks=1000,selected_env_ticks=int(selection.sum()),all_sampled_physics_tick_commands_checked=True,max_step_rad=float(delta.max()),fingers_exact=True),prior_joint17_gates_remain_failed=True,input_action_reward_and_planner_checks_passed=True,physical_admission_passed=False,physical_admission_not_implied=True,actual_input_action_reward_audit=v,selected_cases=selected,planner_replay=replay,
        exact_starts=starts,locked_first8=prefix,unselected_full20_exact_to_best_probe=unchanged,sampled_command_step_rad_per_0p1s=step,
        baseline=summarize(br),best_probe=summarize(bestrows),candidate=summarize(cr),
        lost_best_clean_cases=[i for i,(x,y) in enumerate(zip(bestrows,cr)) if clean(x) and not clean(y)],
        gained_best_clean_cases=[i for i,(x,y) in enumerate(zip(bestrows,cr)) if not clean(x) and clean(y)],
        remaining_target_cases=[i for i in [0,1,2,3,6,7,10,11] if not clean(cr[i])],
        added_physical_cases=[i for i,(x,y) in enumerate(zip(bestrows,cr)) if not any(x['failures'].values()) and any(y['failures'].values())],
        common_clean_effort_vs_best_probe=effort)
    dest=folder/'independent_tracking_audit_31.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k] for k in ['candidate','lost_best_clean_cases','gained_best_clean_cases','remaining_target_cases','added_physical_cases']}))
if __name__=='__main__':main()
