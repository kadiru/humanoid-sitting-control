"""Pre-main smaller-action validation; failed .004-radian evidence stays intact."""
import argparse, traceback
import numpy as np
from positive_slope_common_v3 import *
from positive_slope_validation_v3 import physical, matched, replay, actors
from verify_seat_load_study import evaluation
from run_transfer_assessment import run, status


def reuse_zero(seed):
    """Audit the exact bounded-source change and reuse span-independent zero replay."""
    archive = read(DATA/'early_span_004_archive_03/revision.json')
    check(archive['preserved_exploration_hashes'])
    check({p: h for p, h in archive['original_source_hashes'].items() if '/scripts/' in p})
    scripts = ROOT/'scripts'
    pairs = [
        ('positive_slope_common_v2.py', 'positive_slope_common_v3.py',
         [('SPAN=.004', 'SPAN=.001'), ('run_positive_slope_ppo_v2.py', 'run_positive_slope_ppo_v3.py')]),
        ('run_positive_slope_ppo_v2.py', 'run_positive_slope_ppo_v3.py',
         [('positive_slope_common_v2', 'positive_slope_common_v3'),
          ("'run_positive_slope_ppo_v2.py'", "'run_positive_slope_ppo_v3.py'"), ('default=.004', 'default=.001')]),
        ('positive_slope_validation_v2.py', 'positive_slope_validation_v3.py',
         [('positive_slope_common_v2', 'positive_slope_common_v3')]),
    ]
    hashes = {}
    for old, new, changes in pairs:
        expected = (scripts/old).read_text()
        for before, after in changes:
            assert before in expected
            expected = expected.replace(before, after)
        assert expected == (scripts/new).read_text(), (old, new, 'unexpected semantic change')
        for name in [old, new]: hashes[str(scripts/name)] = sha(scripts/name)
    proof_path = DATA/f'preflight_v2/zero_validation_{seed}.json'
    proof = read(proof_path)
    assert proof['passed'] and proof['all_trace_arrays_exact'] and proof['global_and_indexed_reset_match']
    left, right = Path(proof['left']), Path(proof['right'])
    with np.load(left/'trace.npz') as a, np.load(right/'trace.npz') as b:
        assert set(a.files) == set(b.files)
        for k in a.files: assert np.array_equal(a[k], b[k]), k
    with np.load(left/'episode_decisions.npz') as a, np.load(right/'episode_decisions.npz') as b:
        assert np.array_equal(a['parent_observation'], b['observation'])
        assert np.array_equal(a['parent_action'], b['action'])
    for folder in [left, right]:
        for key, value in read(folder/'manifest.json')['sources'].items():
            assert sha(key) == value, key
        for name in ['trace.npz', 'episode_decisions.npz', 'manifest.json']:
            hashes[str(folder/name)] = sha(folder/name)
    calibration = DATA/'preflight_v2/calibrate/early_calibration.json'
    hashes[str(calibration)] = sha(calibration)
    hashes[str(proof_path)] = sha(proof_path)
    return dict(passed=True, hashes=hashes, seed=seed,
                explanation='Only bound/default and versioned import/source-copy names change. Zero early action remains exactly zero at either span; normalization and reward calibration contain no early action. Same-GPU full global/indexed trace and actual parent inputs/actions rechecked exactly. No new zero simulation claimed.')


def tick_checks(folder, episodes):
    rows = read(folder/'results.json')['episodes']
    with np.load(folder/'every_tick.npz') as z:
        n = z['no_slip'].shape[1]
        assert len(z['t']) == 1000*episodes + 25*(episodes-1)
        for i in range(episodes):
            active = slice(i*1025, i*1025+1000)
            rr = rows[i*n:(i+1)*n]
            for field, metric in [('no_slip', 'slip_ticks'), ('legal', 'illegal_ticks'), ('traction', 'traction_bad_ticks')]:
                assert (~z[field][active]).sum(0).tolist() == [r[metric] for r in rr], (folder, i, field)
    return dict(passed=True, episodes=episodes, active_ticks=1000*episodes*n, warmup_excluded_from_outcomes=25*(episodes-1)*n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    a = ap.parse_args(); seed = SEEDS[a.gpu]
    pre = DATA/'preflight_bounded_03'; pre.mkdir(exist_ok=True)
    state = DATA/f'preflight_bounded_03_status_gpu{a.gpu}.json'
    try:
        assert not (DATA/'ppo_01').exists()
        status(state, 'bounded_source_and_zero_reuse_audit', seed=seed, span=SPAN)
        proof = reuse_zero(seed); write(pre/f'zero_reuse_{seed}.json', proof)
        for name in ['reference', 'parent']:
            physical(DATA/'preflight_v2'/name); evaluation(DATA/'preflight_v2'/name, 18)
        checks = {}
        for label, mode in [(f'explore_{seed}', 'explore'), (f'smoke_{seed}', 'train')]:
            status(state, label, seed=seed, span=SPAN)
            run(pre, label, command(pre, a.gpu, label, mode, seed=seed, updates=2))
            checks[label] = dict(physical=physical(pre/label, 2), ticks=tick_checks(pre/label, 2))
        cp = pre/f'smoke_{seed}/final.pt'; label = f'smoke_eval_{seed}'
        status(state, 'matched_smoke_evaluation', seed=seed, span=SPAN)
        run(pre, label, command(pre, a.gpu, label, 'evaluate', seed=seed, cp=cp))
        physical(pre/label); evaluation(pre/label, 18); matched(DATA/'preflight_v2/reference', pre/label, 9.)
        validation = dict(physical=physical(pre/label), ticks=tick_checks(pre/label, 1),
                          optimizer=replay(pre/f'smoke_{seed}', 2), actor=actors(pre/label, cp), preceding=checks)
        check(proof['hashes']); check(read(DATA/'prior_hashes.json'))
        write(pre/f'validation_{seed}.json', validation)
        status(state, 'complete', seed=seed, span=SPAN, validation=validation)
    except Exception as e:
        status(state, 'failed', error=repr(e), traceback=traceback.format_exc()); raise


if __name__ == '__main__': main()
