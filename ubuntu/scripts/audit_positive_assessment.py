"""Verify the initial positive-only baseline assessment without using holdout."""
import numpy as np
from positive_slope_common import *
from positive_slope_validation import physical
from verify_seat_load_study import evaluation,starts,compare
from verify_posture_credit_study import contacts

def main():
    out=DATA/'assessment/verified_01';out.mkdir(exist_ok=False)
    result=dict(passed=False,paths={},outcomes={},contacts={},starts={},comparisons={},prior_hashes=len(read(DATA/'prior_hashes.json')))
    records={};geoms={}
    for name in ['reference','59111','59117','60111','60117']:
        folder=DATA/'assessment'/name;summary,rows,geometry=evaluation(folder,24)
        outcome=physical(folder,require_clean=False);assert starts(DATA/'assessment/reference',folder)['exact']
        a=np.load(DATA/'assessment/reference/traction_startup.npz');b=np.load(folder/'traction_startup.npz')
        for k in a.files:assert np.array_equal(a[k],b[k]),(name,k)
        result['paths'][name]=str(folder);result['outcomes'][name]=dict(**outcome,independent_evaluation=summary)
        result['contacts'][name]=contacts(folder);result['starts'][name]='Exact physical and414-channel sensor startup'
        records[name]=[dict(x,success=bool(x['success'] and not x['slip_ticks'] and not x['illegal_ticks'] and not any(x['failures'].values()))) for x in rows];geoms[name]=geometry
    for name in records:
        result['comparisons'][name]=compare(records[name],records['reference'],geoms[name])
    check(read(DATA/'prior_hashes.json'));result['passed']=True
    result['scope']='Development assessment only; holdout still closed. No new PPO trained. Former negative-slope failures excluded by user, not relabeled.'
    write(out/'verification.json',result);print({k:{s:v[s] for s in ['stable','clean','physical','slip','illegal']} for k,v in result['outcomes'].items()})
if __name__=='__main__':main()
