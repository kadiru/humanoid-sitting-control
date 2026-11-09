"""Gate the new learning setup on development only; never touch holdout."""
import argparse,time,traceback
from pathlib import Path
from posture_credit_common import *
from run_transfer_assessment import run,status,write

def physical(path,episodes):
    rows=read(path/'results.json')['episodes'];assert len(rows)==episodes*18
    assert all(x['steps']==1000 for x in rows)
    assert all(x['success'] and not any(x['failures'].values()) and x['slip_ticks']==0 and x['illegal_ticks']==0 for x in rows)
    # Every trial starts identically; retain the reference's two prefix flags.
    expected={x['case']:x['traction_bad_ticks'] for x in read(OLD/'env_reference_01/results.json')['episodes']}
    assert all(x['traction_bad_ticks']<=expected[x['case']] for x in rows)
    return dict(passed=True,stable=len(rows),trials=len(rows),strict=sum(x['strict_transfer'] for x in rows))

def waitfile(path):
    while not path.exists():
        for f in DATA.glob('preflight_status_gpu*.json'):
            assert read(f)['phase']!='failed',(f,read(f))
        time.sleep(10)

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();seed=SEEDS[a.gpu]
    state=DATA/f'preflight_status_gpu{a.gpu}.json';folder=DATA/'preflight';folder.mkdir(exist_ok=True)
    try:
        status(state,'waiting_for_development_probes')
        while read(DATA/f'probe_status_gpu{a.gpu}.json')['phase']!='complete':
            assert read(DATA/f'probe_status_gpu{a.gpu}.json')['phase']!='failed'
            time.sleep(10)
        if a.gpu==0:
            status(state,'calibration_and_zero_action_regression')
            run(folder,'calibrate',command(folder,0,'calibrate','calibrate'))
            physical(folder/'calibrate',1)
            import numpy as np
            from verify_seat_load_study import starts
            assert starts(OLD/'env_reference_01',folder/'calibrate')['exact']
            x=np.load(OLD/'env_reference_01/trace.npz');y=np.load(folder/'calibrate/trace.npz')
            for k in x.files:assert np.array_equal(x[k],y[k]),('zero action regression',k)
            write(folder/'zero_action_validation.json',dict(passed=True,physical_and_sensor_start_exact=True,all_saved_trace_arrays_exact=True))
        waitfile(folder/'zero_action_validation.json')
        status(state,'random_persistent_action_validation')
        label=f'explore_{seed}';run(folder,label,command(folder,a.gpu,label,'explore',seed=seed,updates=3));physical(folder/label,3)
        status(state,'two_update_optimizer_smoke')
        label=f'smoke_{seed}';run(folder,label,command(folder,a.gpu,label,'train',seed=seed,updates=2));physical(folder/label,2)
        cp=folder/label/'final.pt';label=f'smoke_eval_{seed}'
        run(folder,label,command(folder,a.gpu,label,'evaluate',checkpoint=cp));physical(folder/label,1)
        status(state,'complete',seed=seed,random_trials=54,smoke_training_trials=36,smoke_final_trials=18)
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
