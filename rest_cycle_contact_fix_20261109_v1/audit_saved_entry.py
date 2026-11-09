"""Check only the first divergence from the saved run; no counterfactual rollout."""
import os,time,resource,signal,json,hashlib
from pathlib import Path
START=time.monotonic();os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def deadline(*_):raise TimeoutError('45_SECOND_ENTRY_AUDIT')
signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,45)
import numpy as np
from contact_gate import ContactGate,Packet
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent
O=R/'outputs/rest_cycle_load_validation_20261109_v1/narrow_load_calibration'
for p,h in json.loads((H/'entry_freeze.json').read_text())['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
plan=json.loads((R/'experiments/rest_cycle_measured_support_20261109_v2/construction/plan.json').read_text())
commands=np.asarray(plan['forward']['commands']);nodes=plan['nodes'];ends=[];begin=0
for node in nodes[1:]:
    indices=np.flatnonzero(np.max(abs(commands[begin:]-node['command']),axis=1)<1e-12)
    assert len(indices);begin+=int(indices[0])+1;ends.append(begin)
hover_node=next(i for i,n in enumerate(nodes) if n.get('normal') and abs(n['gap']-.012)<1e-10)
contact_node=next(i for i,n in enumerate(nodes) if n.get('normal') and abs(n['gap'])<1e-10)
hover,contact=ends[hover_node-1],ends[contact_node-1]
gate=ContactGate(hover,contact)
with np.load(O/'trajectory.npz') as f:z=dict(f)
sg=json.loads((O/'controller_signals.json').read_text())
initial=np.arctan2(z['gravity_pre'][0,0,0],-z['gravity_pre'][0,0,2]);previous_index=0;first=None
for t,row in enumerate(sg):
    g=z['gravity_pre'][t,0]
    # Optimistic tracking admission here isolates the new entry requirement.
    # No load adapter is asserted valid; the old loaded calibration failed.
    packet=Packet(4*t,tuple(float(x) for x in z['qd_pre'][t,0,:23]),
        float(np.arctan2(g[0],-g[2])-initial),float(z['gyro_pre'][t,0,1]),
        row['geometry_valid'],row['geometry_age_s'],True)
    d=gate.update(previous_index,packet)
    if row['path_index']>previous_index and not d['advance']:
        first=dict(control=t+1,time_s=(t+1)*.02,path_index_before=previous_index,
            old_path_index_after=row['path_index'],gate=d,
            meaning='First command the new gate refuses. Later saved states are not a counterfactual trajectory.')
        break
    previous_index=row['path_index']
assert first is not None and first['path_index_before']==hover
meta=json.loads((O/'results.json').read_text());runtime=json.loads((R/'outputs'/H.name/'runtime_inspection/runtime_properties.json').read_text())
kp=np.asarray(runtime['native']['get_dof_stiffnesses'])[0];kd=np.asarray(runtime['native']['get_dof_dampings'])[0]
# Runtime backend order equals imported metadata only after comparing names.
assert runtime['joint_names']==meta['joint_names']
rows=[]
for name in ['left_ankle_roll_joint','right_ankle_roll_joint']:
    j=meta['joint_names'].index(name);e=z['previous'][-250:,0,j]-z['q_pre'][-250:,0,j]
    v=z['qd_pre'][-250:,0,j];tau=z['native_motor_tau_pre'][-250:,0,j]
    rows.append(dict(joint=name,mean_position_error_rad=float(e.mean()),native_motor_effort_Nm=float(tau.mean()),
        endpoint_PD_estimate_Nm=float((kp[j]*e-kd[j]*v).mean()),raw_speed_rad_s=float(v.mean()),
        interpretation='Endpoint PD estimate is approximate; discrepancy is not a corrected sensor or a proved engine cause.'))
result=dict(hover_path_index=hover,contact_path_index=contact,first_divergence=first,
    native_effort_comparison=rows,source_fields_aligned=True,physical_repair_verified=False,
    runtime_loaded_observer_admitted=False,physics_steps=0,wall_s=time.monotonic()-START)
with (R/'outputs'/H.name/'saved_entry_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result));signal.setitimer(signal.ITIMER_REAL,0)
