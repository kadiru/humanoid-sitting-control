"""One bounded24-case physical development probe; no automatic PPO/holdout."""
from pathlib import Path
import datetime,hashlib,json,re,shutil,subprocess,traceback
from low_positive_common import feedback_command
from run_transfer_assessment import run,status
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
O=D/'centered_probe_234';STATE=D/'status_centered_probe_234.json'
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
    for folder,file in [('bias_feedback_checks_v3_207','independent_checks.json'),('centered_runtime_paths_231','results.json'),('centered_geometry_audit_232','independent_audit.json'),('measured_guard_audit_131','independent_audit.json'),('balanced_runtime_preparation_v2_149','preparation.json'),('centered_control_audit_232','independent_audit.json'),('centered_runtime_preparation_233','preparation.json'),('touchdown_schedule_audit_174','independent_audit.json')]:
        a=read(D/folder/file);assert a.get('passed',a.get('all_selected_paths_passed',False))
        for p,h in a['source_sha256'].items():assert sha(p)==h,p
    checks=[]
    for name in ['prior_hashes.json','completed_probes_preservation_230.json','supplemental_preservation_230.json']:
        m=read(D/name);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h]
        checks.append(dict(manifest=name,checked=len(m),mismatches=bad,passed=not bad));assert not bad,bad[:3]
    (D/'prior_recheck_before_centered_234.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2))
    O.mkdir();(O/'frozen_source').mkdir()
    names=['prepare_low_seat_centered_paths_231.py','audit_low_seat_centered_geometry_232.py','render_low_seat_all_attempts_237.py','verify_low_seat_all_attempts_238.py','low_seat_bias_feedback_v2_207.py','check_low_seat_bias_v3_207.py','audit_low_seat_centered_235.py','audit_low_seat_centered_dual_236.py','finalize_low_seat_centered_239.py','low_seat_touchdown_path_175.py','check_low_seat_touchdown_schedule_173.py','low_seat_balance_feedback_160.py','low_seat_bias_runtime_v2_208.py','run_low_seat_centered_233.py','audit_low_seat_centered_control_232.py','prepare_low_seat_centered_runtime_233.py','audit_low_seat_touchdown_schedule_174.py','low_seat_balanced_force_v2_145.py','low_seat_balanced_runtime_v2_148.py','run_low_seat_balanced_v2_148.py','prepare_low_seat_balanced_runtime_v2_149.py','audit_low_seat_balanced_forces_146.py','audit_low_seat_balanced_filter_147.py','low_seat_fixed_foot_plan_126.py','prepare_low_seat_fixed_runtime_127.py','low_seat_gated_command_128.py',
           'audit_low_seat_gated_command_v2_129.py','low_seat_measured_guard_130.py','audit_low_seat_measured_guard_131.py',
           'low_seat_fixed_runtime_132.py','run_low_seat_fixed_132.py','prepare_low_seat_fixed_runtime_133.py',Path(__file__).name,
           'low_seat_dual_view_40.py','low_seat_normal_timing_controller.py','low_seat_load_controller.py','low_seat_contact_controller.py',
           'run_low_positive_feedback_v2.py','positive_slope_runtime_v2.py','run_progressive_seat_ppo.py','foot_wrench_observer.py','run_supported_transition.py']
    files=[R/'scripts'/n for n in names]+[D/'centered_runtime_preparation_233/candidate_config.json',D/'normal_landing3s_guarded_09.json',D/'cooked_model_50/model.json',D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json']
    sources={str(p):sha(p) for p in files}
    for p in files:shutil.copy2(p,O/'frozen_source'/p.name)
    revision=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='PRE-TRAINING centered-stance experiment234. Only measured36cm/+15deg route receives full fixed-foot227 path from current1s sensors: zero lateral root motion and bounded refined roll/pitch/yaw. Other rows use guardednormal3s. Both placements231/232 pass full91-knot geometry/force checks and actual CPU/GPU command/filter/parent fixtures. The new path reduces assumed foot-load asymmetry and ankle-roll command change diagnosed219; causality and actual raw-speed settling remain unproven. Preserve210 proportional/integral gains, antiwindup and authority,175 raw-speed/pitch/gyro quiet guards,130 measured geometry/load guards, frozen parent15s and every physical criterion. This is a pose-path research change, not an execution fix. All earlier failures remain preserved.',
        control='Current encoders/IMU/torques/measured plane only. Current rear fit preferred; selected missing-rear frames may use current down fallback40. No saved sensors in runtime. Cooked model only for planning/clearance; perception raw self-mask unchanged. Stop path progress for unavailable/changed plane, inconsistent foot anchors, clearance<3mm, inadequate finite patch near plane or joint tracking error>.12rad. These are controller guards, not relaxed physical criteria. Assumed zero-gap load progression is capped by estimated foot unloading minus.02BW and never called sensed legal contact. Fixed-foot estimator cannot detect common foot slip.',
        unchanged='All materials, full20/final5 support, tilt40deg AND legal normal seat load>=.05BW, slips/contact-union/physical thresholds, .03 desired joint margin, .1 filter, .15rad/s command cap,14finger commands and frozen59117 weights retained. Parent adds at15s selected and11/15 elsewhere; isolatingprobe omits early61111. No policy promotion.',
        targets='Measured height35.7–36.3cm/slope14.5–15.5 only, two probe rows0/1. Failed227 cases36/+20,37/+16.25,36.5/+17.5 retain guardednormal3s and remain unresolved.',
        first_exact_seconds=1.,unselected_full_exactness='Must be measured; rendered-scene differences are not waived.',
        physical_gpu=1,environments=24,episode_seconds=20.,active_ticks=24000,initial_settling_excluded=600,indexed_warmup=0,optimizer_updates=0,main_training=False,fresh_holdout=False,automatic_followup=False,sources=sources)
    (O/'revision.json').write_text(json.dumps(revision,indent=2))
    baseline=D/'clearance_damped_01/baseline';(O/'baseline').symlink_to(baseline,target_is_directory=True)
    (O/'baseline_reuse.json').write_text(json.dumps(dict(source=str(baseline),simulated_again=False,physical_gpu=1,hashes={p.name:sha(p) for p in baseline.iterdir() if p.is_file()}),indent=2))
    protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md';shutil.copy2(protocol,O/'protocol_before_234.md')
    with protocol.open('a') as f:f.write('\n\nPre-training revision234: '+revision['scope']+' '+revision['control']+' '+revision['unchanged']+' One24-case20sdevelopment probe,24000active+600settling,zerooptimizer/holdout. No automatic retention/PPO. See centered_probe_234/revision.json.\n')
    mutable=read(D/'status.json');shutil.copy2(D/'status.json',O/'status_before_234.json')
    mutable.update(updated_utc=revision['utc'],phase='touchdown_physical_probe_running',physical_jobs_running=True,latest_running_candidate='centered_probe_234',main_training_started=False,fresh_holdout_started=False,repair_complete=False,notes=revision['targets'])
    (D/'status.json').write_text(json.dumps(mutable,indent=2))
    try:
        cmd=feedback_command(O,1,'candidate','zero',D/'banks/probe',D.parent/'low_positive_transfer_01/learning_reference.json',kind='original',seed=63101,start=10.)
        cmd[2]='../scripts/run_low_seat_centered_233.py';cmd+=['--landing-config',str(D/'normal_landing3s_guarded_09.json'),'--fixed-config',str(D/'centered_runtime_preparation_233/candidate_config.json')]
        status(STATE,'running',main_training_started=False,fresh_holdout_started=False);run(O,'candidate',cmd)
        assert all(sha(p)==h for p,h in sources.items())
        status(STATE,'physics_complete_audit_pending',main_training_started=False,fresh_holdout_started=False,repair_complete=False)
        mutable=read(D/'status.json');mutable.update(phase='touchdown_physics_complete_audit_pending',physical_jobs_running=False,latest_running_candidate=None)
        (D/'status.json').write_text(json.dumps(mutable,indent=2))
    except Exception as e:
        status(STATE,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
