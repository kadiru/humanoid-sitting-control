"""Recorded5ms joint-state/PD-telemetry diagnosis. No controller or physics change. Private recorder cache preservation independently audited."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'private_diagnostic_198';O=D/'private_diagnosis_200'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):return dict(min=float(np.min(a)),max=float(np.max(a)),mean=float(np.mean(a)),median=float(np.median(a)))
def main():
    O.mkdir(exist_ok=False);audit=read(D/'private_audit_199/independent_audit.json');assert audit['all_substep_raw_exact'] and all(x['exact'] for x in audit['substep_trace'].values())
    s=np.load(P/'substep_states.npz');meta=read(P/'actuator_metadata.json');names=meta['joint_names'];t=s['t'];q=s['q'];qd=s['qd']
    ii=np.flatnonzero(t>=11.-1e-9);assert len(ii)==401
    fd=np.diff(q.astype(np.float64),axis=0)/np.diff(t)[:,None,None];rows=[]
    for c in range(24):
        joints=[]
        for j,n in enumerate(names):
            v=qd[ii,c,j];vfd=fd[ii-1,c,j];delta=q[ii[-1],c,j]-q[ii[0],c,j]
            integral=float(np.sum(.5*(v[1:]+v[:-1])*np.diff(t[ii])))
            joints.append(dict(joint=n,raw_velocity_rad_s=stats(v),absolute_raw_velocity_rad_s=stats(abs(v)),finite_difference_rad_s=stats(vfd),absolute_finite_difference_rad_s=stats(abs(vfd)),position_range_rad=float(np.ptp(q[ii,c,j])),net_position_change_rad=float(delta),trapezoidal_velocity_integral_rad=integral,integral_minus_position_rad=integral-float(delta),raw_velocity_sign_changes=int(np.sum(v[1:]*v[:-1]<0)),positive_samples=int((v>0).sum()),negative_samples=int((v<0).sum()),substep_phase_mean_velocity=[float(qd[ii[s['substep'][ii]%4==phase],c,j].mean()) for phase in range(4)]))
        rows.append(dict(case=c,joints=joints))
    #Implicit actuator telemetry is computed BEFORE each solver step. Check both
    #alignments, reporting errors rather than asserting it is measured motor torque.
    kp=np.array(meta['stiffness'],np.float32);kd=np.array(meta['damping'],np.float32);limit=np.array(meta['effort_limits'],np.float32)
    pre=kp*(s['command'][1:]-q[:-1])-kd*qd[:-1];post=kp*(s['command'][1:]-q[1:])-kd*qd[1:]
    pre=np.clip(pre,-limit,limit);post=np.clip(post,-limit,limit);telemetry=s['torque'][1:]
    cache=kp*(s['command'][1:]-s['public_q_before'][:-1])-kd*s['public_qd_before'][:-1];cache=np.clip(cache,-limit,limit)
    effort=dict(previous_recorded_public_cache_formula_max_error_Nm=float(abs(cache-telemetry).max()),pre_substep_formula_max_error_Nm=float(abs(pre-telemetry).max()),post_substep_formula_max_error_Nm=float(abs(post-telemetry).max()),formula='clip(Kp*(command-q)-Kd*qd, +/-effort_limit); target velocity/extra effort presumed zero from unchanged action configuration. Residuals reported; not fitted.',semantics=meta['effort_semantics'])
    b=read(P/'balance_feedback_161.json');bias=[]
    for c in [0,1,6,7]:
        r=[x for x in b if x['case']==c and 550<=x['tick']<650];f=[x['feedback'] for x in r]
        bias.append(dict(case=c,samples=len(r),pitch_error_actual_minus_goal_deg=stats(np.rad2deg([x['pitch_rad']-x['target_pitch_rad'] for x in f])),anchor_valid_samples=sum(x['anchor_valid'] for x in f),offset_rad=stats(np.array([x['offset_after'] for x in f])),authority_rad=stats(np.array([x['authority_rad'] for x in f])),estimated_load_factor=stats(np.array([x['estimated_foot_load_fraction'] for x in f])),saturated_samples=int(sum(np.any(np.abs(x['offset_after'])>=np.array(x['authority_rad'])-1e-6) for x in f))))
    original=np.load(D/'touchdown_probe_178/candidate/trace.npz');trace=np.load(P/'trace.npz');divergences={}
    for group,cases in [('all',list(range(24))),('selected',[0,1,6,7])]:
        divergences[group]={}
        for key in ['q','body','command','reference','geometry','torque']:
            aa=trace[key][:,cases];bb=original[key][:130,cases];same=(aa==bb)|(np.isnan(aa)&np.isnan(bb));kk=np.flatnonzero(~same.reshape(130,-1).all(1))
            divergences[group][key]=dict(exact=not bool(len(kk)),first_divergence_s=None if not len(kk) else float(trace['t'][kk[0]]),max_abs=float(abs(aa-bb)[np.isfinite(aa)&np.isfinite(bb)].max()))
    executor=Path('/home/kadir/humanoid-sitting/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_collection.py')
    source=executor.read_text();start=source.index('    def compute(self, dt: float = 0.0)');end=source.index('        for actuator, joint_indices_wp',start);snippet=source[start:end]
    assert snippet.index('self._implicit_executor.launch(self)')<snippet.index('joint_pos = self._control.joint_pos')
    code_witness=dict(path=str(executor),sha256=sha(executor),snippet=snippet,interpretation='The fused implicit executor launches using cached state arrays before this compute function refreshes joint_pos/joint_vel. Recorder198 writes only its private descriptor buffers and independently checks unchanged public bytes/timestamps. Compare the full original prefix and earlier189 failure separately; no installed source changed.')
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,window_s=[11,13],samples_per_case=len(ii),all24_joint_records=rows,prefix_divergence=divergences,telemetry_code_witness=code_witness,selected_feedback=bias,effort_telemetry=effort,exact_prefix_replay=audit['exact_prefix_replay'],physics_ticks=0,optimizer_updates=0,physical_admission=False,interpretation='Direct backend joint velocity,5ms position differences and implicit-drive telemetry are distinct quantities. A persistent mismatch is not proof of a sensor bug, true continuous velocity or a reason to relax a guard. Recorded5ms samples cannot resolve solver-internal iterations. Model/load mismatch and finite proportional feedback can sustain pose bias; causal repair untested. Originalphysical outcomes, reward/effort definitions, materials and criteria remain unchanged.',source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),P/'substep_states.npz',P/'actuator_metadata.json',P/'balance_feedback_161.json',D/'private_audit_199/independent_audit.json']})
    (O/'results.json').write_text(json.dumps(out,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    fig,ax=plt.subplots(3,4,figsize=(15,8),sharex=True)
    for col,c in enumerate([0,1,6,7]):
        j=names.index('right_ankle_roll_joint');ax[0,col].plot(t[ii],qd[ii,c,j],label='backend qd');ax[0,col].plot(t[ii],fd[ii-1,c,j],label='5ms delta q / dt');ax[0,col].set_title(f'Case{c}: right ankle roll')
        ax[1,col].plot(t[ii],1000*(q[ii,c,j]-q[ii[0],c,j]));ax[2,col].plot(t[ii],s['torque'][ii,c,j],label='reported PD effort')
        for a in ax[:,col]:a.grid(alpha=.25)
        ax[2,col].set_xlabel('Time(s)')
    ax[0,0].set_ylabel('Velocity(rad/s)');ax[0,0].legend(fontsize=8);ax[1,0].set_ylabel('Angle change(mrad)');ax[2,0].set_ylabel('PD telemetry(Nm)')
    fig.suptitle('Diagnostic198 private-buffer hover at5ms (see exact replay audit) — diagnostic quantities, no physical repair\nBackend velocity is not substituted by finite differences; torque is approximate implicit-PD telemetry');fig.tight_layout();fig.savefig(O/'substep_velocity.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for c in [0,1,6,7]:
        r=[x for x in b if x['case']==c and 400<=x['tick']<650];tt=np.array([x['tick']*.02 for x in r]);ff=[x['feedback'] for x in r]
        ax[0].plot(tt,np.rad2deg([x['pitch_rad']-x['target_pitch_rad'] for x in ff]),label=f'Case{c}');ax[1].plot(tt,[x['offset_after'][0] for x in ff],label=f'Case{c}')
    ax[0].set_ylabel('Actual - target pitch(deg)');ax[1].set_ylabel('Ankle-pitch feedback(rad)')
    for a in ax:a.grid(alpha=.25);a.set_xlabel('Time(s)');a.legend()
    fig.suptitle('Persistent hover tracking error with unchanged proportional feedback\nNo gain/guard/material change; private-buffer recorder; no physical repair or guard change');fig.tight_layout();fig.savefig(O/'hover_feedback.png',dpi=150);plt.close(fig)
    (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps(dict(effort_telemetry=effort,selected_feedback=bias)),flush=True)
if __name__=='__main__':main()
