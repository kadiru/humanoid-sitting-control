"""Independent saved-sensor replay of the new bounded balance blend."""
import argparse,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);a=p.parse_args();d=a.folder
cfg=json.loads((d/'landing_config.json').read_text());records=json.loads((d/'balance_feedback_audit.json').read_text())
plans=json.loads((d/'rolling_plan_audit.json').read_text());shear=np.load(d/'shear_observer.npz');e=np.load(d/'every_tick.npz')
inertial=json.loads((d/'static_inertial_model.json').read_text());model_weight=sum(inertial['mass'])*9.81
selected_plans={r['case']:r['tick'] for r in plans if r.get('landing_repair_applied')}
previous=np.zeros(len(records[0]['fraction']),dtype=np.float32);max_error=0.;max_step=0.;observed_error=0.
for r in records:
    at=int(np.argmin(abs(shear['t']-r['tick']*.02)))
    assert abs(shear['t'][at]-r['tick']*.02)<1.e-6
    expected_observed=1-shear['filtered'][at,:,:,1].sum(1)/model_weight
    observed=np.array(r['observed_share']);observed_error=max(observed_error,float(abs(expected_observed-observed).max()))
    assert np.allclose(expected_observed,observed,atol=2.e-7,rtol=1.e-6)
    condition=np.max(shear['condition'][at],axis=1)<200
    assert np.array_equal(condition,r['condition_valid'])
    selected=np.array(r['selected'])
    # At tick500 the reference callback is recorded before _observe installs
    # the newly computed plan. The next 0.1-second callback first consumes it.
    expected_selected=np.array([i in selected_plans and r['tick']>selected_plans[i] for i in range(len(previous))])
    assert np.array_equal(selected,expected_selected)
    desired=(np.clip((observed-.05)/.30,0,1)*condition).astype(np.float32)
    if cfg['balance_mode']=='ankle_only':desired[:]=0
    expected=previous+np.clip(desired-previous,-.04,.04);expected[~selected]=0
    actual=np.array(r['fraction'],dtype=np.float32)
    error=float(abs(actual-expected).max());max_error=max(error,max_error)
    assert error<=1.e-7 and np.isfinite(actual).all() and np.min(actual)>=0 and np.max(actual)<=1
    max_step=max(max_step,float(abs(actual-previous).max()));assert max_step<=.0400001
    previous=actual
out=dict(passed=True,mode=cfg['balance_mode'],records=len(records),max_memory_replay_error=max_error,
    max_observed_load_identity_error=observed_error,max_fraction_step=max_step,
    source='Recorded joint-torque/encoder/IMU foot-wrench estimator and static inertial model; no physical force used to reconstruct controller inputs.',
    model_weight_minus_simulator_weight_N=float(model_weight-e['weight'][0,0]),
    tolerance_note='Float32 replay tolerance1e-7; observed-share identity2e-7+1e-6relative. No physical threshold changes.',
    scope='Audits the new balance blend plus recorded condition and selector, not a physical success certificate.')
dest=d/'balance_replay_audit_v2.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2));print(json.dumps(out))
