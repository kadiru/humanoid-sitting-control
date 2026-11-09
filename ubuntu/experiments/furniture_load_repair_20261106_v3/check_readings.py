"""Independent regression and software-contract checks on saved predictions.

No retraining, simulation, score adjustment, or claim that abstention fixes a
physical failure. Coverage is reported separately and cannot pass vacuously.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from load_estimator import QualifiedLoadEstimator, LoadReading, load_feedback

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'furniture_load_estimator_20261106_v2'


def main():
    frozen = json.loads((HERE/'freeze.json').read_text())
    for path, expected in frozen['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    with np.load(PRIOR/'dataset.npz') as z:
        data = dict(z)
    with np.load(PRIOR/'predictions.npz') as z:
        pred = dict(z)
    rows = json.loads((PRIOR/'inputs.json').read_text())
    train = (data['split'] == 0) & data['valid']
    low, high = data['x'][train].min(0), data['x'][train].max(0)
    # Keep the original error bound so the effect of the missing domain gate
    # can be isolated. It is NOT promoted to a furniture-wide bound.
    selector = [0]
    def predictor(x):
        i = selector[0]
        np.testing.assert_array_equal(x, data['x'][i])
        return pred['mean'][i], pred['std'][i]
    estimator = QualifiedLoadEstimator(predictor, low, high, pred['calibration_bound'])
    reports, failures = [], []
    for group, row in enumerate(rows):
        if row['split'] != 'test':
            continue
        ix = np.flatnonzero((data['group'] == group) & data['valid'])
        old_false = new_false = unknown = positives = detected = 0
        reasons = {}
        for i in ix:
            selector[0] = i
            reading = estimator.read(data['x'][i])
            old = pred['mean'][i, 0]-pred['radius'][i, 0] >= .05
            zero = data['y'][i, 0] < .01
            positive = data['y'][i, 0] > .12
            old_false += int(old and zero)
            new_false += int(reading.loaded and zero)
            unknown += int(reading.state == 'unknown')
            positives += int(positive)
            detected += int(positive and reading.loaded)
            if reading.state == 'unknown':
                fraction, search = load_feedback(reading, .25, .3)
                assert fraction == .25 and not search
            if old and zero:
                reasons[reading.reason] = reasons.get(reading.reason, 0)+1
                failures.append(dict(case=row['label'],control=int(i-group*999+1),
                    old_mean_normal_bw=float(pred['mean'][i,0]),
                    true_normal_bw=float(data['y'][i,0]),new_state=reading.state,
                    reason=reading.reason,outside_channels=list(reading.outside_channels)))
        reports.append(dict(case=row['label'],samples=len(ix),old_false_contacts=old_false,
            new_false_contacts=new_false,unknown=unknown,positive_frames=positives,
            detected_positive_frames=detected,loaded_recall=detected/positives if positives else None,
            old_false_contact_disposition=reasons))
    assert sum(r['old_false_contacts'] for r in reports) == 14
    assert sum(r['new_false_contacts'] for r in reports) == 0
    assert len(failures) == 14 and all(r['new_state'] == 'unknown' for r in failures)
    # Unknown and absent/failed sensor data must not become contact or no-contact.
    for packet in [np.zeros(332), np.full(333, np.nan), np.zeros(333)]:
        value = estimator.read(packet)
        assert value.state == 'unknown' and not value.loaded
        assert load_feedback(value, .2, .3) == (.2, False)
    qualified = LoadReading('loaded', np.array([.2,0,0]), np.array([.15,0,0]), np.array([.25,0,0]), 'test')
    fraction, search = load_feedback(qualified, 0., .3)
    assert 0 < fraction <= .008 and not search
    np.savez_compressed(HERE/'sensor_envelope.npz',low=low,high=high,
                        calibration_bound=pred['calibration_bound'])
    result = dict(software_regression_passed=True,old_false_contacts=14,new_false_contacts=0,
        false_contacts_correctly_marked_unknown=14,physical_repairs=0,PPO_updates=0,
        simulation_steps=0,all_case_reports=reports,false_contact_records=failures,
        furniture_wide_estimator_admitted=False,
        limitation='Domain checking repairs false confidence, not missing force information. Unknowns remain unknown; loaded recall and five-object coverage are not waived. No new controller launch is admitted by this result alone.',
        calibration_defect='Original calibration contains only old flat/inclined cases and armchair leans, no other furniture. Marginal ensemble intervals are not valid under this shift.')
    (HERE/'regression.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['all_case_reports','false_contact_records']}),flush=True)


if __name__ == '__main__':
    main()
