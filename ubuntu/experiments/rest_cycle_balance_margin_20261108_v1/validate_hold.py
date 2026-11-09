"""Saved-label interface regression only; no actor signal or physical rollout.

Seeds the HOLD stage deliberately to isolate its contract. This does not prove
that any recorded trajectory reached that stage with a sensor-only controller.
"""
import os, time, signal, resource, json, hashlib, sys
from pathlib import Path
START=time.monotonic()
os.environ['CUDA_VISIBLE_DEVICES']=''
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def deadline(*args):raise TimeoutError('45S_TOTAL_SAVED_VALIDATION_CAP')
signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,45.)
import numpy as np
from stages import RestCycle, Sensors, Stage
H=Path(__file__).resolve().parent

def main():
    contract=json.loads((H/'hold_contract.json').read_text())
    for name,h in contract['files'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    rows=[]
    for item in contract['cases']:
        with np.load(item['trajectory']) as z:
            # Previously recorded evaluator labels, used only for this offline
            # interface test. No runtime adapter is created or certified here.
            good=z['good'][:,0].astype(bool)
            load=z['support_bw'][:,0]
            raw_quiet=np.max(abs(z['physical_joint_velocity'][:,0,:23]),axis=1)<=.05
            assert len(good)==1000 and good[750:1000].all()
            c=RestCycle();c.stage=Stage.HOLD
            old_counter=0;labels=[]
            for i in range(1000):
                old_counter=old_counter+1 if good[i] and raw_quiet[i] and load[i]>=.05 else 0
                labels.append(c.update(Sensors(map_valid=True,tracking_ok=True,
                    quiet=bool(raw_quiet[i]),rest_stability_qualified=bool(good[i]),
                    load_lower_bw=float(load[i]),load_upper_bw=float(load[i])))['stage'])
            assert c.rest_window_passed and c.stage==Stage.HOLD
            assert all(s=='hold' for s in labels[750:1000])
            prior_hold=c.hold_ticks
            out=c.update(Sensors(map_valid=True,tracking_ok=True,quiet=False,
                rest_stability_qualified=True,load_lower_bw=float(load[-1]),load_upper_bw=float(load[-1])))
            assert out['stage']=='unload' and not out['advance_primitive']
            unchanged_clean=bool(good[750:1000].all() and z['no_slip'].all()
                and z['legal'].all() and not z['failure'].any())
            assert unchanged_clean==item['previous_clean']
            rows.append(dict(case=item['id'],saved_samples=1000,old_raw_quiet_hold_s=old_counter*.02,
                corrected_qualified_hold_s=prior_hold*.02,raw_quiet_final5s_samples=int(raw_quiet[750:1000].sum()),
                original_clean_label_unchanged=unchanged_clean,first_unload_tick=c.tick,
                motion_advance_with_nonquiet_unload=out['advance_primitive']))
    result=dict(passed=True,rows=rows,classification='SAVED_LABEL_INTERFACE_REGRESSION_ONLY',
        truth_labels_used_only_offline=True,precontact_reachability_validated=False,
        actor_rest_sensor_qualified=False,complete_physical_cycles=0,
        physics_steps=0,PPO_updates=0,wall_s=time.monotonic()-START)
    (H/'hold_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':main()
