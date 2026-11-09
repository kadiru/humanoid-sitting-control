"""Finish admission, then launch only the authorized fixed-budget sitting pilot."""
import time,json,shutil,traceback
from pathlib import Path
import numpy as np
from verify_repaired_sitting import evaluation,plan,starts,sha,load
from run_transfer_assessment import write,status
import run_repaired_sitting_comparison as study

pre=study.PRE;out=pre/'admission_02'
try:
    while not (out/'zero_55217.exit').exists():time.sleep(15)
    assert int((out/'zero_55217.exit').read_text())==0
    report=dict(passed=False,selected_training_parent=55211,rejected_training_parent=55217,evaluations={},plans={},starts={},zero_identity={})
    for label,folder in [('reference',pre/'reference'),('parent_55211',pre/'parent_55211'),('zero_55211',pre/'zero_55211'),('parent_55217',pre/'parent_55217'),('zero_55217',out/'zero_55217')]:
        m=study.read(folder/'manifest.json');cp=m['args'].get('checkpoint')
        ev,_=evaluation(folder,18,Path(cp) if cp else None);report['evaluations'][label]=ev;report['plans'][label]=plan(folder);report['starts'][label]=starts(study.PREVIOUS/'reference',folder)
        assert ev['physical_failures']==0
        if label in ['reference','parent_55211','zero_55211']:assert ev['successes']==18
    for seed,zero in [(55211,pre/'zero_55211'),(55217,out/'zero_55217')]:
        parent=pre/f'parent_{seed}'
        for file in ['startup.npz','trace.npz']:
            a=np.load(parent/file);b=np.load(zero/file);assert a.files==b.files
            assert all(np.array_equal(a[k],b[k]) for k in a.files),(seed,file)
        assert study.read(parent/'results.json')==study.read(zero/'results.json')
        initial=load(zero/'initial.pt');assert initial['actor_state_dict']['mlp.4.weight'].count_nonzero()==0 and initial['actor_state_dict']['mlp.4.bias'].count_nonzero()==0
        report['zero_identity'][str(seed)]=dict(startup_trace_and_all_results_exact=True)
    assert study.read(pre/'reset_isolation/reset_probe.json')['passed']
    for path,h in study.read(pre/'source_freeze.json').items():assert sha(path)==h,path
    prior=study.read(study.DATA/'prior_preservation.json')['hashes']
    for path,h in prior.items():assert sha(path)==h,path
    report.update(passed=True,prior_hashes=len(prior),reset_isolation=True,unseen_sitting_holdout_used=False)
    write(out/'independent_validation.json',report)
    write(out/'verification.json',dict(passed=True,selected_training_parent=55211,reference_successes=18,selected_parent_successes=18,excluded_parent_successes=16,excluded_parent_physical_failures=0,zero_identity=True,criteria_unchanged=True,original_failed_admission_preserved=True))
    shutil.copy2(__file__,out/'admission_source.py');status(out/'status.json','admitted_common_parent')
    study.worker()
except Exception as e:
    status(out/'launch_status.json','failed',error=repr(e),traceback=traceback.format_exc());raise
