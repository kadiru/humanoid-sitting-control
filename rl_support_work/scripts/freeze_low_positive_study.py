"""Freeze only after both software/physics/optimizer preflights finish."""
import time,shutil
from low_positive_common import *
from positive_slope_common_v5 import write,sha,check
from validate_low_positive_feedback_v3 import validate

def main():
    root=DATA/'ppo_01';assert not root.exists();proposal=read(DATA/'learning_proposal.json')
    assert read(DATA/'feedback_zero_v2_status.json')['phase']=='passed'
    hashes=dict(read(OLD/'ppo_01/protocol.json')['hashes']);prior=read(DATA/'prior_hashes.json')
    check(prior);check(read(DATA/'rejected_repairs_preservation.json'))
    for seed in proposal['seeds']:
        assert read(DATA/f'preflight_recovery_status_{seed}.json')['phase']=='passed'
        pre=DATA/'preflight_01'/f'seed_{seed}'
        for label in ['calibration','exploration','smoke','smoke_evaluation']:
            validate(pre/label)
            m=read(pre/label/'manifest.json');hashes.update(m['sources'])
            for name,h in m['bank_hashes'].items():hashes[str(Path(m['args']['bank'])/name)]=h
    for name in ['run_low_positive_study.py','freeze_low_positive_study.py','launch_low_positive_study_v2.py','low_positive_common.py','validate_low_positive_feedback_v2.py','validate_low_positive_feedback_v3.py','run_low_positive_preflight_v2.py','verify_low_positive_study.py','finalize_low_positive_study.py','feedback_posture_ppo.py','feedback_slip_reward.py','run_low_positive_feedback_v2.py','run_low_positive_preflight.py']:
        f=ROOT/'scripts'/name;hashes[str(f)]=sha(f)
    for f in (DATA/'banks').rglob('*'):
        if f.is_file():hashes[str(f)]=sha(f)
    hashes[str(DATA/'learning_reference.json')]=sha(DATA/'learning_reference.json')
    for folder in ['preflight_01','feedback_zero_01','zero_validation_recovery_01','input_audit_recovery_02','documentation_history_01']:
        for f in (DATA/folder).rglob('*'):
            if f.is_file():prior[str(f)]=sha(f)
    prior.update(read(DATA/'rejected_repairs_preservation.json'))
    check(hashes);check(prior);root.mkdir()
    # One optimizer process avoids resetting its RNG between blocks. The8s
    # prospective test is evaluation only; all16 training updates start10s.
    frozen=dict(proposal,frozen=True,freeze_timestamp=time.time(),blocks=1,updates_per_block=16,
        active_per_seed=448000,indexed_warmup_per_seed=10500,total_per_seed=458500,total_main=917000,initial_settling_excluded_total=1400,
        control_stages_seconds=[10.],prospective_evaluation_only_seconds=[8.],checkpoint_selection_updates=[4,8,12,16],
        selection='Gate-passing10s development checkpoints ranked by clean count, physical failures, then common-clean whole20 effort. Test the best once at8s; use8s only if its independent gate passes. Explicit parent fallback otherwise. All training remains at10s.',
        pre_main_revision='Replace four optimizer-process blocks with one uninterrupted16-update run; preserve model, Adam and random stream. Earlier8s is an explicitly untrained prospective evaluation. No main had started when revised.',
        hashes=hashes,prior_hashes=prior)
    write(root/'protocol.json',frozen);(root/'protocol.sha256').write_text(sha(root/'protocol.json')+'\n');shutil.copy2(DATA/'learning_proposal.json',root/'preserved_block_proposal.json')
    print(dict(frozen=True,sources=len(hashes),protected=len(prior),total_main=917000))

if __name__=='__main__':main()
