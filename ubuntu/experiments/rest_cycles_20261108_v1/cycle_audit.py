"""Independent full-cycle scoring; simulator truth is evaluation only."""
import numpy as np


def audit(z,task,*,evidence_kind):
    if evidence_kind not in ('synthetic_fixture','recorded_simulation'):raise ValueError('UNSPECIFIED_EVIDENCE')
    required=['q','qd','command','desired','failure','no_slip','legal','good','rest_load_bw','pitch_error','gyro_y','standing_height_error','stage','time_s']
    if not all(k in z for k in required):raise ValueError('INCOMPLETE_CYCLE_RECORD')
    if not all(k in z for k in ['initial_command','standing_q','lower','upper']):raise ValueError('MISSING_RESET_OR_ACTION_BOUNDS')
    n=len(z['q']);complete=1000<n<=2000 and all(len(z[k])==n for k in required)
    if not complete:return dict(passed=False,reason='incomplete_record',physical_success=False)
    q=np.asarray(z['q']);command=np.asarray(z['command']);stages=list(z['stage'])
    order=[s for i,s in enumerate(stages) if i==0 or s!=stages[i-1]]
    expected=['observe','position','contact','transfer','hold','unload','stand','done']
    finite=all(np.isfinite(np.asarray(z[k])).all() for k in required if k!='stage')
    before=np.vstack([z['initial_command'],command[:-1]])
    checks=dict(finite=finite,full_phase_order=order==expected,
       timestamps=bool(np.allclose(z['time_s'],.02*np.arange(1,n+1),rtol=0,atol=1e-9)),
       no_physical_failure=not np.asarray(z['failure']).any(),
       no_slip=bool(np.asarray(z['no_slip']).all()),legal_contacts=bool(np.asarray(z['legal']).all()),
       original20s_rest=bool(np.asarray(z['good'])[750:1000].all()),
       hold_stage_in_original_window=all(s=='hold' for s in stages[750:1000]),
       no_early_exit=not any(s in ('unload','stand','done') for s in stages[:1000]),
       return_without_support=bool((np.asarray(z['rest_load_bw'])[-100:]<.01).all()),
       standing_joint_error=bool((abs(q[-100:,:23]-np.asarray(z['standing_q'])[:23])<=.05).all()),
       standing_quiet=bool((abs(np.asarray(z['qd'])[-100:,:23])<=.05).all()),
       standing_pitch=bool((abs(np.asarray(z['pitch_error'])[-100:])<=np.deg2rad(.5)).all()),
       standing_gyro=bool((abs(np.asarray(z['gyro_y'])[-100:])<=.02).all()),
       standing_height=bool((abs(np.asarray(z['standing_height_error'])[-100:])<=.03).all()),
       rate=bool((abs(command-before)<=.003+2e-7).all()),
       desired_margin=bool(((np.asarray(z['desired'])>=np.asarray(z['lower'])+.03-2e-7)&(np.asarray(z['desired'])<=np.asarray(z['upper'])-.03+2e-7)).all()),
       filter=bool(np.max(abs(command[:,:23]-(before[:,:23]+np.clip(.1*(np.asarray(z['desired'])-before[:,:23]),-.003,.003))))<=2e-7),
       fingers_fixed=bool(np.array_equal(command[:,23:],np.broadcast_to(np.asarray(z['initial_command'])[23:],command[:,23:].shape))))
    if task=='upright_lean':checks['upright_knees']=bool((abs(q[750:1000][:,[11,12]])<=np.deg2rad(35)).all())
    elif task!='sitting':raise ValueError('UNSPECIFIED_TASK')
    return dict(passed=all(checks.values()),checks=checks,physical_success=all(checks.values()) and evidence_kind=='recorded_simulation',evidence_kind=evidence_kind,
                original_rest_window_seconds=20,cycle_seconds=n*.02,stand_settle_seconds=2)
