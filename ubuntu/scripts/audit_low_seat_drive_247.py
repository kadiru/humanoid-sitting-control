"""Independent accounting/source/prefix/substep checks for private link/drive recording246."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
import torch
from posture_episode_ppo import PosturePPO
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'drive_diagnostic_246';B=D/'centered_probe_234/candidate';O=D/'drive_audit_247'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def diff(a,b):
    assert a.shape==b.shape,(a.shape,b.shape)
    same=(a==b)|(np.isnan(a)&np.isnan(b));ok=np.isfinite(a)&np.isfinite(b)
    return dict(exact=bool(same.all()),changed=int((~same).sum()),max_finite_difference=float(abs(a[ok].astype(np.float64)-b[ok].astype(np.float64)).max()) if ok.any() else None)
def main():
    O.mkdir(exist_ok=False);torch.set_num_threads(2)
    protocol=read(P/'diagnostic_protocol.json');account=read(P/'diagnostic_accounting.json');meta=read(P/'actuator_metadata.json')
    assert account['physics_substeps']==2600 and account['active_ticks']==15600 and account['initial_settling_excluded']==600
    assert account['elapsed_s']==13. and account['physics_dt']==.005 and not account['complete20s_episodes'] and not account['optimizer_updates'] and not account['fresh_holdout']
    assert not account['recording_control_use'] and account['scene_update_callbacks']==2600 and account['substep_records']==1001
    assert all(sha(p)==h for p,h in protocol['sources'].items())
    trace=np.load(P/'trace.npz');original=np.load(B/'trace.npz');s=np.load(P/'substep_states.npz');assert len(trace['t'])==130
    replay={k:diff(trace[k],original[k][:130]) for k in ['t','body','q','command','geometry','reference','torque']}
    starts={}
    for n in ['startup.npz','traction_startup.npz','bridge_startup.npz']:
        a=np.load(P/n);b=np.load(B/n);starts[n]={k:diff(a[k],b[k]) for k in a.files if k in b.files}
    ticks=np.load(P/'every_tick.npz');old_ticks=np.load(B/'every_tick.npz')
    tick_replay={k:diff(ticks[k],old_ticks[k][:650]) for k in ticks.files if k in old_ticks.files and ticks[k].shape[0]==650}
    sensor=np.load(P/'actual_sensor_frames.npz')['sensors'];old=np.load(B/'sensor_inputs.npz')['sensors'][0,:130]
    sensor_replay=diff(sensor,old);assert sensor.shape==(130,24,414) and np.isfinite(sensor).all()
    assert np.array_equal(s['substep'],np.arange(1600,2601)) and np.max(abs(s['t']-s['substep']*.005))==0
    assert all(np.isfinite(s[k]).all() for k in s.files)
    idx=np.flatnonzero(s['substep']%20==0);ti=s['substep'][idx]//20-1
    substep_trace={k:diff(s[k][idx],trace[k][ti]) for k in ['q','command','torque']}
    raw=read(P/'touchdown_raw_176.json');points={(x['case'],x['tick']):x for x in raw};raw_checks=[]
    for j in np.flatnonzero(s['substep']%4==0):
        tick=int(s['substep'][j]//4)
        for c in [0,1]:
            r=points.get((c,tick))
            if r is None:continue
            for key in ['q','qd']:raw_checks.append(diff(s[key][j,c],np.array(r[key],np.float32)))
    assert all(x['exact'] for x in substep_trace.values()) and all(x['exact'] for x in raw_checks)
    recorder=read(P/'recorder_metadata.json');assert recorder['records']==1001 and recorder['private_query_destinations']
    status=read(P/'step_status.json');assert status['completed_control_steps']==130 and status['active_transitions']==15600 and status['physics_substeps']==2600
    assert not (P/'execution_failure.txt').exists() and len(trace['t'])==130 and abs(float(trace['t'][-1])-13.)<1e-9
    order=np.array(recorder['user_to_backend']);assert np.array_equal(np.sort(order),np.arange(37))
    reorder_checks={k:diff(s[k],s[k+'_backend'][:,:,order]) for k in ['q','qd']}
    cache_checks={k:diff(s[k+'_before'],s[k+'_after']) for k in ['public_q','public_qd','cache_timestamp']}
    assert all(x['exact'] for x in reorder_checks.values()) and all(x['exact'] for x in cache_checks.values())
    assert recorder['public_pointers']==recorder['implicit_kernel_state_pointers']
    assert not set(recorder['public_pointers'])&set(recorder['backend_pointers'])
    assert recorder['every_read_public_cache_bytes_and_timestamps_exact'] and recorder['every_backend_query_nonoverlapping_public_storage']

    #Read-only recorder must not be granted an equality waiver.
    exact=all(x['exact'] for x in replay.values()) and sensor_replay['exact'] and all(x['exact'] for x in tick_replay.values()) and all(x['exact'] for z in starts.values() for x in z.values())
    names=meta['joint_names'];finger=[i for i,n in enumerate(names) if any('_'+x+'_joint' in n for x in ['zero','one','two','three','four','five','six'])];assert len(finger)==14
    assert np.all(s['command'][:,:,finger]==s['command'][0,:,finger].T[None])
    assert all(x=='ImplicitActuator' for x in meta['actuator_types'].values())
    action=read(P/'parent_actions.json');assert len(action)==1 and action[0]['step']==110
    obs=np.array(action[0]['observation'],np.float32);assert np.array_equal(obs,sensor[110])
    cmd=read(D/'centered_probe_234/candidate.command.json');ck=cmd[cmd.index('--checkpoint')+1];assert sha(ck)==protocol['parent_sha256']
    parent=PosturePPO(np.zeros(414),np.ones(414),0);parent.load(ck)
    with torch.no_grad():pa=parent.choose(torch.from_numpy(obs),False)[0].numpy()
    err=float(abs(pa-np.array(action[0]['action'],np.float32)).max());assert np.allclose(pa,np.array(action[0]['action'],np.float32),atol=1e-5,rtol=1e-5)
    launch=np.load(P/'implicit_launches.npz');count=len(launch['t_before']);assert count==1000 and np.array_equal(launch['substep_before'],np.arange(1600,2600))
    assert recorder['all_allocated_timestamped_caches_unchanged'] and recorder['cache_count_min']>=2
    body_check=diff(s['link_transforms'][idx],trace['body'][ti])
    target_check=diff(s['dof_position_targets'],s['command'])
    # Independent float64 algebra of exactly captured pre-launch inputs.
    # Roundoff bound is arithmetic only: 8 float32 eps times the sum of
    # absolute products/input effort; no physical or equality tolerance changes.
    v={k:launch[k].astype(np.float64) for k in ['target_q','target_qd','target_effort','cached_q','cached_qd','kp','kd','limits']}
    pterm=v['kp']*(v['target_q']-v['cached_q']);dterm=v['kd']*(v['target_qd']-v['cached_qd']);expected=pterm+dterm+v['target_effort']
    bound=8*np.finfo(np.float32).eps*(abs(v['kp']*v['target_q'])+abs(v['kp']*v['cached_q'])+abs(v['kd']*v['target_qd'])+abs(v['kd']*v['cached_qd'])+abs(v['target_effort'])+1)
    computed_error=abs(expected-launch['computed_effort']);applied_error=abs(np.clip(expected,-v['limits'],v['limits'])-launch['applied_effort'])
    formula_passed=bool((computed_error<=bound).all() and (applied_error<=bound).all())
    # After each exact recorded launch, its telemetry must persist into that
    # solver step's sample. This is a specified one-step alignment, not a fit.
    li=launch['substep_before']+1-1600
    telemetry_check=diff(launch['applied_effort'],s['torque'][li])
    exact=bool(exact and body_check['exact'] and target_check['exact'] and telemetry_check['exact'] and formula_passed)
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),diagnostic_checks_passed=exact,exact_prefix_replay=exact,physical_admission=False,repair_complete=False,accounting=account,postflight_completion_checked_from_saved_arrays=True,post_context_completion_marker_present=(P/'completion.json').exists(),source_hashes_unchanged=True,prefix_trace=replay,prefix_ticks=tick_replay,exact_starts=starts,actual414=sensor_replay,substep_trace=substep_trace,substep_raw_checks=len(raw_checks),all_substep_raw_exact=True,backend_to_public_order_replay=reorder_checks,public_cache_unchanged=cache_checks,recorder_metadata=recorder,fixed_fingers=len(finger),actual_parent_cpu_action_max_error=err,parent_tolerances=dict(atol=1e-5,rtol=1e-5),actuator_types=meta['actuator_types'],effort_semantics=meta['effort_semantics'],interpretation='Diagnostic replay only. Every direct backend query checks separate storage and unchanged public cache bytes/timestamps. Full prefix must also replay exactly; no equality waiver, sensor substitution, guard or controller change. No new20s success, PPO or holdout.',additional_drive_checks=dict(body_transforms=body_check,backend_position_targets=target_check,launch_to_poststep_telemetry=telemetry_check,implicit_launch_records=count,float64_formula_passed=formula_passed,formula_max_computed_error_Nm=float(computed_error.max()),formula_max_applied_error_Nm=float(applied_error.max()),formula_bound_definition='8 float32 eps * (sum absolute input products +1)',maximum_velocity_target_rad_s=float(abs(v['target_qd']).max()),maximum_feedforward_effort_Nm=float(abs(v['target_effort']).max())),source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),P/'diagnostic_protocol.json',P/'substep_states.npz',P/'actual_sensor_frames.npz',P/'actuator_metadata.json',P/'implicit_launches.npz',P/'live_stage_settings.json']})
    (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps({k:out[k] for k in ['diagnostic_checks_passed','exact_prefix_replay','substep_raw_checks','physical_admission']}),flush=True)
if __name__=='__main__':main()
