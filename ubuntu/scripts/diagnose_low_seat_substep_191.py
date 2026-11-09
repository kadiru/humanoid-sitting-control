"""Recorded5ms joint-state/PD-telemetry diagnosis. No controller or physics change."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'substep_diagnostic_189';O=D/'substep_diagnosis_191'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):return dict(min=float(np.min(a)),max=float(np.max(a)),mean=float(np.mean(a)),median=float(np.median(a)))
def main():
    O.mkdir(exist_ok=False);audit=read(D/'substep_audit_v2_190/independent_audit.json');assert audit['diagnostic_checks_passed']
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
    effort=dict(pre_substep_formula_max_error_Nm=float(abs(pre-telemetry).max()),post_substep_formula_max_error_Nm=float(abs(post-telemetry).max()),formula='clip(Kp*(command-q)-Kd*qd, +/-effort_limit); target velocity/extra effort presumed zero from unchanged action configuration. Residuals reported; not fitted.',semantics=meta['effort_semantics'])
    b=read(P/'balance_feedback_161.json');bias=[]
    for c in [0,1,6,7]:
        r=[x for x in b if x['case']==c and 550<=x['tick']<650];f=[x['feedback'] for x in r]
        bias.append(dict(case=c,samples=len(r),pitch_error_actual_minus_goal_deg=stats(np.rad2deg([x['pitch_rad']-x['target_pitch_rad'] for x in f])),anchor_valid_samples=sum(x['anchor_valid'] for x in f),offset_rad=stats(np.array([x['offset_after'] for x in f])),authority_rad=stats(np.array([x['authority_rad'] for x in f])),estimated_load_factor=stats(np.array([x['estimated_foot_load_fraction'] for x in f])),saturated_samples=sum(np.any(np.abs(x['offset_after'])>=np.array(x['authority_rad'])-1e-6) for x in f)))
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,window_s=[11,13],samples_per_case=len(ii),all24_joint_records=rows,selected_feedback=bias,effort_telemetry=effort,exact_prefix_replay=True,physics_ticks=0,optimizer_updates=0,physical_admission=False,interpretation='Direct backend joint velocity,5ms position differences and implicit-drive telemetry are distinct quantities. A persistent mismatch is not proof of a sensor bug, true continuous velocity or a reason to relax a guard. Recorded5ms samples cannot resolve solver-internal iterations. Model/load mismatch and finite proportional feedback can sustain pose bias; causal repair untested. Originalphysical outcomes, reward/effort definitions, materials and criteria remain unchanged.',source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),P/'substep_states.npz',P/'actuator_metadata.json',P/'balance_feedback_161.json',D/'substep_audit_v2_190/independent_audit.json']})
    (O/'results.json').write_text(json.dumps(out,indent=2))
    fig,ax=plt.subplots(3,4,figsize=(15,8),sharex=True)
    for col,c in enumerate([0,1,6,7]):
        j=names.index('right_ankle_roll_joint');ax[0,col].plot(t[ii],qd[ii,c,j],label='backend qd');ax[0,col].plot(t[ii],fd[ii-1,c,j],label='5ms delta q / dt');ax[0,col].set_title(f'Case{c}: right ankle roll')
        ax[1,col].plot(t[ii],1000*(q[ii,c,j]-q[ii[0],c,j]));ax[2,col].plot(t[ii],s['torque'][ii,c,j],label='reported PD effort')
        for a in ax[:,col]:a.grid(alpha=.25)
        ax[2,col].set_xlabel('Time(s)')
    ax[0,0].set_ylabel('Velocity(rad/s)');ax[0,0].legend(fontsize=8);ax[1,0].set_ylabel('Angle change(mrad)');ax[2,0].set_ylabel('PD telemetry(Nm)')
    fig.suptitle('Unchanged178 hover recorded at5ms — diagnostic quantities, no physical repair\nBackend velocity is not substituted by finite differences; torque is approximate implicit-PD telemetry');fig.tight_layout();fig.savefig(O/'substep_velocity.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for c in [0,1,6,7]:
        r=[x for x in b if x['case']==c and 400<=x['tick']<650];tt=np.array([x['tick']*.02 for x in r]);ff=[x['feedback'] for x in r]
        ax[0].plot(tt,np.rad2deg([x['pitch_rad']-x['target_pitch_rad'] for x in ff]),label=f'Case{c}');ax[1].plot(tt,[x['offset_after'][0] for x in ff],label=f'Case{c}')
    ax[0].set_ylabel('Actual - target pitch(deg)');ax[1].set_ylabel('Ankle-pitch feedback(rad)')
    for a in ax:a.grid(alpha=.25);a.set_xlabel('Time(s)');a.legend()
    fig.suptitle('Persistent hover tracking error with unchanged proportional feedback\nNo gain/guard/material change and no new success claim');fig.tight_layout();fig.savefig(O/'hover_feedback.png',dpi=150);plt.close(fig)
    (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps(dict(effort_telemetry=effort,selected_feedback=bias)),flush=True)
if __name__=='__main__':main()
