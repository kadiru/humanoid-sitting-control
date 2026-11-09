"""One bounded24-case physical development probe; no automatic PPO/holdout."""
from pathlib import Path
import datetime,hashlib,json,re,shutil,subprocess,traceback
from low_positive_common import feedback_command
from run_transfer_assessment import run,status
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
O=D/'roll_feedback_probe_273';STATE=D/'status_roll_feedback_probe_273.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def main():
    assert not O.exists() and not STATE.exists()
    assert shutil.disk_usage(D).free>20*1024**3
    procs=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines()
    assert not [p for p in procs if re.search(r'^\s*\d+\s+\S*python\S* .*scripts/(run|refine|optimize|audit|diagnose|schedule|check|bound|screen)_low_seat_\S+\.py',p)]
    for folder,file in [('bias_feedback_checks_v3_207','independent_checks.json'),('centered_runtime_paths_231','results.json'),('centered_geometry_audit_232','independent_audit.json'),('measured_guard_audit_131','independent_audit.json'),('balanced_runtime_preparation_v2_149','preparation.json'),('roll_control_audit_v3_271','independent_audit.json'),('roll_runtime_preparation_272','preparation.json'),('touchdown_schedule_audit_174','independent_audit.json')]:
        a=read(D/folder/file);assert a.get('passed',a.get('all_selected_paths_passed',False))
        for p,h in a['source_sha256'].items():assert sha(p)==h,p
    checks=[]
    for name in ['prior_hashes.json','completed_probes_preservation_266.json','supplemental_preservation_266.json']:
        m=read(D/name);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h]
        checks.append(dict(manifest=name,checked=len(m),mismatches=bad,passed=not bad));assert not bad,bad[:3]
    (D/'prior_recheck_before_roll_273.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2))
    O.mkdir();(O/'frozen_source').mkdir()
    names=['low_seat_roll_feedback_269.py','low_seat_roll_runtime_270.py','diagnose_low_seat_roll_279.py','audit_low_seat_contact_velocity_268.py','prepare_low_seat_centered_paths_231.py','audit_low_seat_centered_geometry_232.py','render_low_seat_all_attempts_276.py','verify_low_seat_all_attempts_277.py','low_seat_bias_feedback_v2_207.py','check_low_seat_bias_v3_207.py','audit_low_seat_roll_274.py','audit_low_seat_roll_dual_275.py','finalize_low_seat_roll_278.py','low_seat_touchdown_path_175.py','check_low_seat_touchdown_schedule_173.py','low_seat_balance_feedback_160.py','low_seat_bias_runtime_v2_208.py','run_low_seat_roll_270.py','audit_low_seat_roll_control_v3_271.py','prepare_low_seat_roll_runtime_v3_272.py','audit_low_seat_touchdown_schedule_174.py','low_seat_balanced_force_v2_145.py','low_seat_balanced_runtime_v2_148.py','run_low_seat_balanced_v2_148.py','prepare_low_seat_balanced_runtime_v2_149.py','audit_low_seat_balanced_forces_146.py','audit_low_seat_balanced_filter_147.py','low_seat_fixed_foot_plan_126.py','prepare_low_seat_fixed_runtime_127.py','low_seat_gated_command_128.py',
           'audit_low_seat_gated_command_v2_129.py','low_seat_measured_guard_130.py','audit_low_seat_measured_guard_131.py',
           'low_seat_fixed_runtime_132.py','run_low_seat_fixed_132.py','prepare_low_seat_fixed_runtime_133.py',Path(__file__).name,
           'low_seat_dual_view_40.py','low_seat_normal_timing_controller.py','low_seat_load_controller.py','low_seat_contact_controller.py',
           'run_low_positive_feedback_v2.py','positive_slope_runtime_v2.py','run_progressive_seat_ppo.py','foot_wrench_observer.py','run_supported_transition.py']
    files=[R/'scripts'/n for n in names]+[D/'roll_runtime_preparation_272/candidate_config.json',D/'normal_landing3s_guarded_09.json',D/'cooked_model_50/model.json',D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json']
    sources={str(p):sha(p) for p in files}
    for p in files:shutil.copy2(p,O/'frozen_source'/p.name)
    revision=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='PRE-TRAINING bounded raw ankle-roll velocity-feedback experiment273. Keep234 current-input complete centered fixed-foot path,207 pitch/hover integral,160 approach balance,175 quiet/normal descent and130 guards. ONLY two measured36/+15 rows gain raw-roll damping at first stationary12mm hover and thereafter while current qd/geometry/load valid. Gain .025s is installed ankleKd/Kp=4/160, not a fitted contact-truth offset or gain sweep. Rollcap .01rad lies INSIDE unchanged .06rad per-ankle absolute pitch-plus-roll authority; original207 pitch priority stays exact. Offset rate .0375rad/s reserves .1125nominal on roll, retaining global .15. Invalid current measurements remove contribution at bounded rate; no roll integral. Parent15s stays additive. Offline267/268 show opposing normal-direction point velocities with small separation and pose differences; these are not bad-sensor or sole-cause proofs. This test asks whether direct measured negative velocity feedback improves actual settling/support, not whether static wrench arithmetic passes.',
        control='Current encoders/IMU/torques/measured plane only. Current rear fit preferred; selected missing-rear frames may use current down fallback40. No saved sensors in runtime. Cooked model only for planning/clearance; perception raw self-mask unchanged. Stop path progress for unavailable/changed plane, inconsistent foot anchors, clearance<3mm, inadequate finite patch near plane or joint tracking error>.12rad. These are controller guards, not relaxed physical criteria. Assumed zero-gap load progression is capped by estimated foot unloading minus.02BW and never called sensed legal contact. Fixed-foot estimator cannot detect common foot slip.',
        unchanged='All materials, full20/final5 support, tilt40deg AND legal normal seat load>=.05BW, slips/contact-union/physical thresholds, .03 desired joint margin, .1 filter, .15rad/s command cap,14finger commands and frozen59117 weights retained. Parent adds at15s selected and11/15 elsewhere; isolatingprobe omits early61111. No policy promotion.',
        targets='Measured height35.7–36.3cm/slope14.5–15.5 only, two probe rows0/1. Failed227 cases36/+20,37/+16.25,36.5/+17.5 retain guardednormal3s and remain unresolved.',
        first_exact_seconds=1.,unselected_full_exactness='Must be measured; rendered-scene differences are not waived.',
        physical_gpu=1,environments=24,episode_seconds=20.,active_ticks=24000,initial_settling_excluded=600,indexed_warmup=0,optimizer_updates=0,main_training=False,fresh_holdout=False,automatic_followup=False,sources=sources)
    (O/'revision.json').write_text(json.dumps(revision,indent=2))
    baseline=D/'clearance_damped_01/baseline';(O/'baseline').symlink_to(baseline,target_is_directory=True)
    (O/'baseline_reuse.json').write_text(json.dumps(dict(source=str(baseline),simulated_again=False,physical_gpu=1,hashes={p.name:sha(p) for p in baseline.iterdir() if p.is_file()}),indent=2))
    protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md';shutil.copy2(protocol,O/'protocol_before_273.md')
    with protocol.open('a') as f:f.write('\n\nPre-training revision273: '+revision['scope']+' '+revision['control']+' '+revision['unchanged']+' One24-case20sdevelopment probe,24000active+600settling,zerooptimizer/holdout. No automatic retention/PPO. See roll_feedback_probe_273/revision.json.\n')
    mutable=read(D/'status.json');shutil.copy2(D/'status.json',O/'status_before_273.json')
    mutable.update(updated_utc=revision['utc'],phase='roll_feedback_physical_probe_running',physical_jobs_running=True,latest_running_candidate='roll_feedback_probe_273',main_training_started=False,fresh_holdout_started=False,repair_complete=False,notes=revision['targets'])
    (D/'status.json').write_text(json.dumps(mutable,indent=2))
    try:
        cmd=feedback_command(O,1,'candidate','zero',D/'banks/probe',D.parent/'low_positive_transfer_01/learning_reference.json',kind='original',seed=63101,start=10.)
        cmd[2]='../scripts/run_low_seat_roll_270.py';cmd+=['--landing-config',str(D/'normal_landing3s_guarded_09.json'),'--fixed-config',str(D/'roll_runtime_preparation_272/candidate_config.json')]
        status(STATE,'running',main_training_started=False,fresh_holdout_started=False);run(O,'candidate',cmd)
        assert all(sha(p)==h for p,h in sources.items())
        status(STATE,'physics_complete_audit_pending',main_training_started=False,fresh_holdout_started=False,repair_complete=False)
        mutable=read(D/'status.json');mutable.update(phase='roll_feedback_physics_complete_audit_pending',physical_jobs_running=False,latest_running_candidate=None)
        (D/'status.json').write_text(json.dumps(mutable,indent=2))
    except Exception as e:
        status(STATE,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
