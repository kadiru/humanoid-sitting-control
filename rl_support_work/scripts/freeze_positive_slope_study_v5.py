"""Freeze a bounded new study only after independent clean-domain preflight."""
import time
from positive_slope_common_v5 import *
from positive_slope_validation_v4 import physical,matched,replay,actors
from verify_seat_load_study import evaluation
from run_positive_bounded_preflight import tick_checks
from run_positive_pulse8_preflight import zero_check8

def main():
    pre=DATA/'preflight_pulse8_05';study=DATA/'ppo_01';assert not study.exists()
    admission={}
    for name in ['reference','parent']:
        admission[name]=physical(DATA/'preflight_v2'/name);evaluation(DATA/'preflight_v2'/name,18)
    for gpu,seed in enumerate(SEEDS):
        assert read(DATA/f'preflight_pulse8_05_status_gpu{gpu}.json')['phase']=='complete'
        assert read(pre/f'zero_validation_{seed}.json')==zero_check8(pre/f'zero_pulse_{seed}',gpu)
        physical(pre/f'zero_pulse_{seed}',2);tick_checks(pre/f'zero_pulse_{seed}',2)
        admission[str(seed)]={}
        probe=DATA/f'pulse_timing_probe_05/explore8_{seed}';assert read(DATA/f'pulse_timing_probe_05/validation_{seed}.json')['passed']
        admission[str(seed)]['exploration']=dict(physical=physical(probe,2),ticks=tick_checks(probe,2),startup=matched(DATA/'preflight_v2/reference',probe,8.))
        for label,count in [(f'smoke_{seed}',2),(f'smoke_eval_{seed}',1)]:
            admission[str(seed)][label]=dict(physical=physical(pre/label,count),ticks=tick_checks(pre/label,count))
        admission[str(seed)]['CPU_optimizer_replay']=replay(pre/f'smoke_{seed}',2)
        admission[str(seed)]['actor_audit']=actors(pre/f'smoke_eval_{seed}',pre/f'smoke_{seed}/final.pt')
        evaluation(pre/f'smoke_eval_{seed}',18);matched(DATA/'preflight_v2/reference',pre/f'smoke_eval_{seed}',8.)
    prior=read(DATA/'prior_hashes.json');check(prior)
    for name in ['assessment','repair_01','repair_02','repair_03','repair_04','repair_05','repair_06','preflight','preflight_v2','preflight_bounded_03','preflight_pulse_04','pulse_timing_probe_05','preflight_pulse8_05','pulse9_archive_05','early_span_001_persistent_archive_04','early_span_004_archive_03','reset_recovery_01','pre_main_revision_01','pre_main_revision_02','repair_close_recovery_01','diagnosis_export_01','repair_validation_export_02','repair_video_qa_02','export_recovery_01']:
        for p in (DATA/name).rglob('*'):
            if p.is_file():prior[str(p)]=sha(p)
    hashes=dict(read(OLD/'ppo_01/protocol.json')['hashes'])
    names=['positive_slope_common_v5.py', 'run_positive_pulse8_preflight.py', 'probe_positive_pulse_timing.py', 'run_positive_slope_study_v5.py', 'verify_positive_slope_study_v5.py', 'freeze_positive_slope_study_v5.py', 'launch_positive_slope_study_v6.py', 'finalize_positive_slope_study_v5.py', 'report_positive_slope_study_v5.py','positive_slope_common_v4.py', 'run_positive_slope_ppo_v4.py', 'positive_slope_validation_v4.py', 'run_positive_pulse_preflight.py', 'run_positive_slope_study_v4.py', 'verify_positive_slope_study_v4.py', 'freeze_positive_slope_study_v4.py', 'launch_positive_slope_study_v5.py', 'finalize_positive_slope_study_v4.py', 'report_positive_slope_study_v4.py','positive_slope_common_v3.py','run_positive_slope_ppo_v3.py','positive_slope_validation_v3.py','run_positive_bounded_preflight.py','run_positive_slope_study_v3.py','verify_positive_slope_study_v3.py','freeze_positive_slope_study_v3.py','launch_positive_slope_study_v4.py','finalize_positive_slope_study_v3.py','positive_slope_common.py','positive_slope_common_v2.py','positive_slope_planner.py','positive_slope_planner_v2.py','positive_slope_runtime_v2.py','positive_timed_reference.py','run_positive_parent_v2.py','run_positive_slope_ppo_v2.py','positive_slope_validation_v2.py','run_positive_preflight_v2.py','run_positive_slope_study_v2.py','verify_positive_slope_study.py','freeze_positive_slope_study.py','launch_positive_slope_study.py','launch_positive_slope_study_v3.py','resume_positive_preflight_v2.py','finalize_positive_slope_study.py','report_positive_slope_study.py','render_positive_slope_video_v3.py','verify_positive_slope_video.py']
    files=[ROOT/'scripts'/n for n in names]+[ROOT/'docs/POSITIVE_SLOPE_PROTOCOL.md',DATA/'reservation.json',DATA/'pulse_revision_04.json',DATA/'pulse8_revision_05.json',DATA/'exploration_failure_diagnosis_02.json',DATA/'selected_reference_v2.json',DATA/'training_admission_revision.json',DATA/'preflight_v2/calibrate/early_calibration.json']
    files+=list((DATA/'banks').glob('*/*'))+[checkpoint(x) for x in [59111,59117,60111,60117]]
    for p in files:
        assert p.is_file(),p
        hashes[str(p)]=sha(p)
    # Include every baseline/preflight manifest dependency without dropping older frozen sources.
    for p in [*pre.glob('*/manifest.json'),* (DATA/'repair_06').glob('*/manifest.json')]:
        for key,value in read(p)['sources'].items():assert sha(key)==value;hashes[key]=value
    protocol=dict(admitted=True,timestamp=time.time(),seeds=SEEDS,parent=59117,stages=STAGES,initial_start_s=8.,block_updates=[3]*4,active_per_seed=216000,warmup_per_seed=3600,transitions_per_seed=219600,total_transitions=439200,decisions_per_seed=216,initial_settle_total_excluded=3600,hashes=hashes,prior_hashes=prior,preflight=admission,source_files=names,
      contract='414 sensor/reference channels;3 symmetric hip/knee/ankle offsets bounded0.001rad one-second smooth approach pulse from8/6s,returned exactly to zero before touchdown; reference controls gap until parent handoff; frozen59117 takes over at11s on retained originals,15s on repaired5s transfers;17other residuals zero,14fingers fixed. Feedback retained. Truth is labels/reset only.',
      reward='Unclipped complete20s all37joint torque squared centered by action-independent matched parent development return; physical20/unsupported10/slip5/contact5/addedmargin2 penalties. Eight fullbatch PPO epochs,std0.6,meanKL0.02rollback; fresh early actor/critic/Adam preserved across blocks.',
      gate='All18 admitted cases full20s stable support,zero all-tick slip/unwanted contact/physical failure; whole20 torque2 <= matched repaired-parent+0.1%, final5 <= parent+1%, both below repaired reference. Conservative.90 warning kept/penalized; legacy.6 ratio is not actual static friction capacity.',
      selection='Earliest passing development takeover then whole20/final5 effort; explicit repaired-parent fallback. No policy promotion. No progression if current/prospective gate fails.',
      banks=dict(training=18,development=24,holdout=24),stress='36cm/+15,+20 and40cm/+20 excluded from training, normalization, calibration and selection; six development trials retained as explicit known failures. Training subset chosen before new optimizer training, not a heldout bank.',
      preflight_experience='All physical probes, zero/reset episodes, random exploration and two-update smokes separate from main; actual contracts retained and independently summed in report.',
      caveats=['Only positive slopes; descending toward feet','One-second bounded approach pulse, not continuous PPO takeover or autonomous motion discovery','Two seeds and repeated nearby rigid slabs','Withheld bank reserved before new simulation, no holdout for learning/selection','Torque proxy, not energy or global optimum','Static robot friction0.5,seat/floor0.8; dynamic0.5/0.6. Legacy force ratio uses0.6; no static-capacity claim from that ratio. Actual slip/contact definitions unchanged'])
    check(hashes);study.mkdir();write(study/'protocol.json',protocol);(study/'protocol.sha256').write_text(sha(study/'protocol.json')+'\n')
    print(dict(admitted=True,frozen=len(hashes),protected=len(prior),budget=439200))
if __name__=='__main__':main()
