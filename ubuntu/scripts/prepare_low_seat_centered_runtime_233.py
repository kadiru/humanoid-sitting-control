"""No-physics integration tests of132 callback ordering and current-input guards."""
from pathlib import Path
from types import SimpleNamespace,ModuleType
import ast,datetime,hashlib,inspect,json,sys,textwrap
import numpy as np
import torch
from humanoid_sitting.shared_frame import BodyModel
from low_seat_touchdown_path_175 import TouchdownPath as GatedCommandPath
from low_seat_measured_guard_130 import MeasuredGuard
import low_seat_fixed_runtime_132 as runtime
import low_seat_bias_runtime_v2_208 as revised
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'centered_runtime_preparation_233'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Parent:
    def reference(self):return self.fake_reference.clone()
class Transition:
    def _observe(self,force=False):pass
def main():
    O.mkdir(exist_ok=False)
    for name,file in [('centered_geometry_audit_232','independent_audit.json'),('centered_control_audit_232','independent_audit.json'),('measured_guard_audit_131','independent_audit.json')]:
        a=read(D/name/file);assert a['passed']
        for p,h in a['source_sha256'].items():assert sha(p)==h,p
    config=read(D/'centered_runtime_paths_231/candidate_config.json')
    config.update(cooked_model=str(D/'cooked_model_50/model.json'),inertia=str(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json'))
    (O/'candidate_config.json').write_text(json.dumps(config,indent=2))
    source_path=Path('scripts/run_supported_transition.py');source=source_path.read_text();tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TransitionEnv')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_observe')
    observe=textwrap.dedent('\n'.join(source.splitlines()[fn.lineno-1:fn.end_lineno]))
    # Replace dependencies only in this isolated software test. No simulation
    # module is imported and no environment/reset/step call is made.
    fake=ModuleType('run_supported_transition');fake.TransitionEnv=Transition
    Transition._observe.__globals__.update(np=np,torch=torch)
    sys.modules['run_supported_transition']=fake
    old_getsource=inspect.getsource;old_normal=runtime.normal_install
    inspect.getsource=lambda obj:observe if obj is Transition._observe else old_getsource(obj)
    runtime.normal_install=lambda *a,**kw:None
    harness=SimpleNamespace(FeedbackEnv=Parent,parent_harness=SimpleNamespace(build=lambda:None),base=SimpleNamespace())
    try:revised.install(harness,{},config)
    finally:inspect.getsource=old_getsource;runtime.normal_install=old_normal
    m=BodyModel(config['cooked_model']);plan=read(D/'centered_runtime_paths_231/case_0.json')['plan']
    # Reuse the newly validated231 centered plans; no repeat optimizer or planner run.
    exact=[0,1]
    assert read(D/'centered_control_audit_232/independent_audit.json')['passed']
    tr=np.load(D/'normal_landing3s_06/candidate/trace.npz');ss=np.load(D/'normal_landing3s_06/candidate/sensor_inputs.npz')['sensors'];at=int(np.argmin(abs(tr['t']-1)))
    geo=tr['geometry'][at,0];q=tr['q'][at,0];g=ss[0,10,0,339:342];command=tr['command'][at,0]
    env=harness.FeedbackEnv.__new__(harness.FeedbackEnv);env.fixed_ready=True;env.balance_ready=True;env.balance_records=[];env.touchdown_records=[];env.cooked_model=m;env.planning_enabled=True;env.t=50;env.fixed_observed_tick=50
    engine=GatedCommandPath(plan,m);guard=MeasuredGuard(m,plan,geo)
    env.fixed_states={0:dict(engine=engine,guard=guard,start=50,tick=50,target=command.copy(),plan=plan)}
    env.warming=torch.zeros(2,dtype=torch.long);env.device='cpu';env.geometry=np.tile(geo,(2,1));env.fixed_guard_records=[]
    env.fake_reference=torch.tensor(np.tile(command,(2,1)),dtype=torch.float32);env.fake_reference[1,0]+=.001
    tensor=lambda v:SimpleNamespace(torch=torch.tensor(v,dtype=torch.float32))
    env.robot=SimpleNamespace(data=SimpleNamespace(joint_pos=tensor(np.tile(q,(2,1))),projected_gravity_b=tensor(np.tile(g,(2,1))),joint_vel=tensor(np.zeros((2,37))),root_ang_vel_b=tensor(np.zeros((2,3)))))
    W=sum(read(config['inertia'])['mass'])*9.81;env.wrench_observer=SimpleNamespace(weight=W)
    env.filtered_wrench=np.zeros((2,2,3));env.filtered_wrench[:,:,1]=W/2
    env.shear_records=[dict(condition=np.ones((2,2))*20)]
    before=env.reference();assert engine.position==0 and np.array_equal(before[1].numpy(),env.fake_reference[1].numpy())
    env.t=51;env.fixed_observed_tick=50;unchanged=env.reference();assert torch.equal(unchanged,before)
    env.fixed_observed_tick=51;first=env.reference();position=engine.position;assert position>0
    for _ in range(8):assert torch.equal(env.reference(),first) and engine.position==position
    env.t=52;assert torch.equal(env.reference(),first) and engine.position==position
    env.fixed_observed_tick=52;second=env.reference();assert engine.position>position
    position=engine.position;env.t=53;env.fixed_observed_tick=53;env.geometry[0,12]=.6
    held=env.reference();assert engine.position==position and not env.fixed_guard_records[-1]['progress_allowed']
    assert len(env.touchdown_records)==3
    assert all(r['tick'] in [51,52,53] for r in env.touchdown_records)
    assert len(env.balance_records)==3 and env.balance_records[-1]['tick']==53
    assert np.max(abs(env.fixed_states[0]['balance_feedback'].offset))<=.06+1e-7
    for _ in range(8):assert torch.equal(env.reference(),held) and len(env.balance_records)==3
    assert np.array_equal(held[1].numpy(),env.fake_reference[1].numpy()) and np.array_equal(held[0,23:].numpy(),command[23:])
    assert len(env.fixed_guard_records)==3 and not env.fixed_guard_records[-1]['progress_allowed']
    # Parent routing consumes only explicit reference-plan flags, not labels.
    from positive_slope_runtime_v2 import parent_decision_steps
    routing=parent_decision_steps(SimpleNamespace(num_envs=2,device='cpu',plan_audit=[dict(case=0,positive_candidate_applied=True)]))
    assert routing.tolist()==[150,110]
    paths=[Path(__file__).resolve(),Path('scripts/low_seat_bias_runtime_v2_208.py').resolve(),Path('scripts/run_low_seat_centered_233.py').resolve(),Path('scripts/low_seat_touchdown_path_175.py').resolve(),Path('scripts/low_seat_fixed_runtime_132.py').resolve(),Path('scripts/run_low_seat_fixed_132.py').resolve(),source_path.resolve(),Path('scripts/low_seat_dual_view_40.py').resolve(),Path('scripts/low_seat_normal_timing_controller.py').resolve(),Path('scripts/positive_slope_runtime_v2.py').resolve(),O/'candidate_config.json']
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,
             scope='No-physics current-input208 bounded-bias integration; unchanged175 measured-settling. Reuses newly validated231 centered plans; no new planner calculation. Nominal path pauses while bounded measured balance remains active. Current observation must finish before reference progress; repeated reference callbacks never advance twice. Stale geometry holds nominal progression; bounded IMU feedback remains active. Unselected rows/fingers exact in this software test. Actual physical input, prefix and retention checks still pending.',
             reused_current_input_validated_plan_cases=exact,new_measured_balance_feedback=True,nominal_rate_reserve_rad_s=.1125,feedback_rate_reserve_rad_s=.0375,current_observation_before_progress=True,repeated_callback_idempotent=True,stale_geometry_pauses=True,
             unselected_mock_reference_exact=True,fixed_fingers_exact=True,parent_selected_absolute_handoff_s=15.,
             source_insertion_reversible=True,physics_ticks=0,optimizer_updates=0,physical_admission=False,
             source_sha256={str(p):sha(p) for p in paths})
    (O/'preparation.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'}),flush=True)
if __name__=='__main__':main()
