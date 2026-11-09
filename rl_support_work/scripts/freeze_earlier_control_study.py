"""Admit only validated development behavior; freeze before main learning."""
import time
from earlier_control_common import *
from run_earlier_control_preflight import physical
from verify_earlier_control_study import replay,actor_checks
from verify_seat_load_study import evaluation,starts
import numpy as np

def main():
    pre=PREFLIGHT;study=DATA/'ppo_01';assert not study.exists()
    admission={}
    for gpu,seed in enumerate(SEEDS):
        assert read(DATA/f'preflight_status_gpu{gpu}.json')['phase']=='complete'
        zero=read(pre/f'zero_validation_{seed}.json');assert zero['passed'] and zero['matched_reset_all_arrays_exact']
        base=OLD/f'ppo_01/seed_{PARENTS[seed]}/development_final'
        admission[str(seed)]={}
        for label,count in [(f'explore_{seed}',2),(f'smoke_{seed}',2),(f'smoke_eval_{seed}',1)]:
            admission[str(seed)][label]=physical(pre/label,count,base)
        smoke=pre/f'smoke_{seed}';evaluated=pre/f'smoke_eval_{seed}'
        admission[str(seed)]['optimizer_replay']=replay(smoke,seed,2)
        admission[str(seed)]['actor_check']=actor_checks(evaluated,seed,smoke/'final.pt',10.)
        evaluation(evaluated,18);assert starts(base,evaluated)['exact']
        a=np.load(base/'trace.npz');b=np.load(evaluated/'trace.npz');mask=a['t']<=10.
        for k in ['q','body','command','reference','torque','geometry']:assert np.array_equal(a[k][mask],b[k][mask])
    prior=read(DATA/'prior_hashes.json');check(prior)
    for name in ['preflight','preflight_bounded_02','early_span_004_archive_01','admission_revision_02','preflight_recovery_01','video_smoke_01','video_smoke_02','video_smoke_03','video_grid_smoke_01','video_grid_smoke_02','video_full_validation_01']:
        for p in (DATA/name).rglob('*'):
            if p.is_file():prior[str(p)]=sha(p)
    for pattern in ['preflight_worker_resumed_gpu*.log','preflight_bounded_worker_gpu*.log','earlier_control_launch_01.log','earlier_control_finalize_01.log','prior_integrity_preflight.json','video_*sources*.json']:
        for p in DATA.glob(pattern):prior[str(p)]=sha(p)
    old=read(OLD/'ppo_01/protocol.json');hashes=dict(old['hashes'])
    names=['earlier_control_common.py','prepare_earlier_control.py','run_earlier_control_ppo.py','run_earlier_control_preflight.py','probe_parent_indexed_reset.py','earlier_control_gate.py','run_earlier_control_study.py','verify_earlier_control_study.py','freeze_earlier_control_study.py','launch_earlier_control_study.py','finalize_earlier_control_study.py','report_earlier_control_study.py','render_earlier_control_video.py','verify_earlier_control_video.py']
    files=[ROOT/'scripts'/n for n in names]+[ROOT/'docs/EARLIER_CONTROL_PROTOCOL.md',DATA/'reservation.json']+list((DATA/'wide_bank_01').iterdir())+[DATA/f'preflight/calibrate_{s}/early_calibration.json' for s in SEEDS]
    for p in files:hashes[str(p)]=sha(p)
    protocol=dict(admitted=True,timestamp=time.time(),seeds=SEEDS,parents=PARENTS,stages=STAGES,block_updates=[6]*4,active_per_seed=432000,warmup_per_seed=9000,transitions_per_seed=441000,total_transitions=882000,decisions_per_seed=432,initial_settle_total_excluded=3600,
      hashes=hashes,prior_hashes=prior,source_files=names,preflight=admission,
      preflight_experience=dict(calibration_active=72000,exploration_active=72000,smoke_active=72000,smoke_eval_active=36000,indexed_warmup=2700,initial_settle=3600,matched_parent_recovery_active=72000,matched_parent_recovery_warmup=900,matched_parent_recovery_initial_settle=900,rejected_004_probe_active=72000,rejected_004_probe_warmup=900,rejected_004_probe_initial_settle=900),
      contract='414 sensor/reference inputs; three early offsets bounded .001rad until11s; frozen parent chooses .008rad resting offsets; 14fixed fingers; repaired reference feedback retained',
      selection='Development only, earliest passing time then whole20 effort then final5 effort. Full-budget final always reported; explicit parent fallback.',
      bank_counts=dict(development=18,retention=16,wide=32),video='All final comparison episodes, success and failure labeled, recorded-state schematic under60s; training exploration excluded',
      caveats=['Reference-assisted; no autonomous standing-motion discovery','Two seeds and repeated rigid slabs; limited generalization','Retention bank previously tested; wide bank reserved before all new simulation','Explicit pre-main admission revision permits one conservative warning tick in exploration/smoke; progression/selection retain no-added-margin requirement','Conservative margin warnings remain distinct from physical slipping','Torque proxy, not measured energy/global optimum'])
    study.mkdir();write(study/'protocol.json',protocol);(study/'protocol.sha256').write_text(sha(study/'protocol.json')+'\n')
    print(dict(admitted=True,frozen=len(hashes),protected=len(prior),budget=882000))
if __name__=='__main__':main()
