"""Independent exact regression against an identical frozen-parent reset sequence."""
import json
import sys
from pathlib import Path
import numpy as np


def exact_arrays(reference, candidate):
    assert set(reference) == set(candidate), 'ARRAY_KEYS'
    for key in reference:
        assert reference[key].shape == candidate[key].shape, ('SHAPE', key)
        assert np.array_equal(reference[key], candidate[key], equal_nan=True), ('EXACT_RESET_IDENTITY', key)


def unit():
    # A matching two-episode trace may differ between fresh and indexed starts.
    first = np.arange(6., dtype=float).reshape(3, 2)
    repeated = np.concatenate([first, np.zeros((1, 2)), first + .001])
    reference = {'q': repeated, 't': np.arange(7.)}
    exact_arrays(reference, {k: v.copy() for k, v in reference.items()})
    failures = 0
    for changed in [
        {'q': np.concatenate([first, np.zeros((1, 2)), first]), 't': np.arange(7.)},
        {'q': repeated[:-1], 't': np.arange(7.)},
        {'q': repeated, 't': np.arange(7.) + .1},
        {'q': repeated},
    ]:
        try:
            exact_arrays(reference, changed)
        except AssertionError:
            failures += 1
    assert failures == 4
    print('RESET_COMPARATOR_UNIT_PASS: matching sequence accepted; drift, shape, time and missing-key errors rejected')


def audit(root):
    root = Path(root)
    parent, zero = root/'parent_reset', root/'calibrate'
    counts = {}
    for filename in ['startup.npz', 'traction_startup.npz', 'trace.npz']:
        with np.load(parent/filename) as a, np.load(zero/filename) as b:
            exact_arrays(dict(a), dict(b))
            counts[filename] = len(a.files)
            if filename == 'trace.npz':
                assert len(a['t']) == 405
    a = np.load(parent/'applied_offsets.npz')['offset']
    b = np.load(zero/'applied_offsets.npz')['offsets']
    assert a.shape == (400, 24, 3) and b.shape == (400, 3, 24, 3)
    assert np.count_nonzero(b[:, 0]) == 0
    assert np.array_equal(a, b[:, 1]) and np.array_equal(a, b[:, 2])
    for folder in [parent, zero]:
        rows = json.loads((folder/'results.json').read_text())['episodes']
        assert len(rows) == 48
        assert all(r['steps'] == 1000 and r['duration_s'] == 20 for r in rows)
    result = dict(passed=True, exact_arrays=counts, trace_samples=405,
                  episodes_per_case=2, compared_command_samples=400,
                  matched_fresh_and_indexed_reset_sequences=True,
                  fresh_equals_indexed_required=False,
                  tolerance_relaxed=False, physical_reset_changed=False,
                  optimizer_updates=0, physical_repair_claimed=False)
    (root/'reset_identity_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    unit() if len(sys.argv) == 1 else audit(sys.argv[1])
