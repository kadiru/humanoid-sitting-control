"""Independent saved-array audit of whole-path timing necessities and422 algebra.
No model/root/optimizer invocation. No conditional timing bound is global.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime,math
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'budget_preparation_audit_423'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False);r=json.loads((D/'budget_preparation_422/preparation.json').read_text());assert r['passed']
    for p,h in r['source_input_sha256'].items():assert sha(p)==h,p
    a=dict(np.load(D/'budget_preparation_422/fixture.npz'));f=dict(np.load(D/'monotone_preparation_410/fixture.npz'));b=dict(np.load(D/'window_path_400/initial.npz'));s=dict(np.load(D/'direct_path_418/structural.npz'));e=dict(np.load(D/'direct_path_418/event_clock.npz'));t=dict(np.load(D/'budget_preparation_422/saved_timing.npz'));report=json.loads((D/'budget_preparation_422/saved_timing.json').read_text())
    req=[];rows=[]
    for k in range(114):
        vals=[.02]
        for j in range(16):
            dc=float(s['C'][k+1,j]-s['C'][k,j]);rate=float(b['rates'][j]-200*e['ulp'][j]);up=float(b['command_hi'][j]-s['C'][k+1,j]-32*e['ulp'][j]);low=float(s['C'][k+1,j]-b['command_lo'][j]-32*e['ulp'][j]);assert min(up,low,rate)>0
            vals.extend([abs(dc)/rate,.18*dc/up,-.18*dc/low])
        if 99<=k<104:vals.append(float((s['Y'][k-1,54]-s['Y'][k,54])/.006))
        req.append(max(vals))
    req=np.array(req);timing_error=float(abs(req-t['required']).max());assert timing_error<1e-12
    bounds=[]
    for g,(start,end) in enumerate([(0,16),(16,64),(64,99),(99,104),(104,114)]):
        total=math.fsum(req[start:end]);ticks=max(math.ceil(total/.02),101 if g==3 else end-start+1);bounds.append(ticks);assert abs(total-report['groups'][g]['conditional_minimum_time_s'])<1e-12 and ticks==report['groups'][g]['conditional_event_tick_requirement']
    assert sum(bounds)==report['event_movement_tick_lower_bound'] and abs(math.fsum(req)+.5-report['with_minimum_synthetic_quiet_s'])<1e-12
    hard=np.array([not(12212<=i<19508 or i==19759 or i>=20284) for i in range(27708)]);assert np.array_equal(hard,a['hard'])
    rows=np.flatnonzero(a['keep']);m=len(rows);J=np.zeros((m+1574,899));values=[]
    for j,i in enumerate(rows):
        values.append(f['iq'][i]/f['scale'][i]+(0. if hard[i] else float(a['slack'])));J[j,:894]=f['I'][i,:894]/f['scale'][i]
        for g in range(4):J[j,894+g]=(f['I'][i,894+g]-f['I'][i,898])/f['scale'][i]
        J[j,-1]=0. if hard[i] else 1.
    low=[];high=[]
    for idx in f['shape']:
        k,c=divmod(int(idx),9);col=[2,3,21,22,16,17,18,19,20][c];low.append(b['lower'][k,col]);high.append(b['upper'][k,col])
    low=np.array(low);high=np.array(high);assert np.array_equal(low,a['lo']) and np.array_equal(high,a['hi'])
    values=np.r_[values,f['v'][:786]-low,high-f['v'][:786],f['z'][-1]-11.,21.-f['z'][-1]]
    J[m:m+786,:894]=f['G'][:786,:894];J[m+786:m+1572,:894]=-f['G'][:786,:894]
    for g in range(4):J[m:m+786,894+g]=f['G'][:786,894+g]-f['G'][:786,898];J[m+786:m+1572,894+g]=-J[m:m+786,894+g]
    J[-2,894:898]=-1.;J[-1,894:898]=1.
    assert np.array_equal(values,a['values']) and np.array_equal(J,a['jacobian'])
    assert np.array_equal(a['lower'][:894],np.load(D/'direct_preparation_416/fixture.npz')['lower']) and np.array_equal(a['upper'][:894],np.load(D/'direct_preparation_416/fixture.npz')['upper'])
    assert np.array_equal(a['budget_lo'],[88,189,241,101,11]) and np.array_equal(a['budget_hi'],[98,199,251,106,21])
    assert np.array_equal(a['lower'][894:],a['budget_lo'][:4]) and np.array_equal(a['upper'][894:],a['budget_hi'][:4])
    dot=float(abs(J@a['direction']-a['separate_direction']).max());assert dot<2e-12
    r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,timing_bound_replay_error_s=timing_error,conditional_fixed_waveform_movement_ticks=sum(bounds),conditional_fixed_waveform_with_quiet_ticks=sum(bounds)+25,all28889rows_exact=True,jacobian_shape=list(J.shape),budget_conservation_derivative_exact=True,continuous_domain_unchanged=True,algebra_directional_error=dot,physics_ticks=0,policy_optimizer_updates=0,physical_admission=False,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),D/'budget_preparation_422/preparation.json',D/'budget_preparation_422/fixture.npz',D/'budget_preparation_422/saved_timing.npz',D/'budget_preparation_422/saved_timing.json']})
    (O/'independent_audit.json').write_text(json.dumps(r,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
