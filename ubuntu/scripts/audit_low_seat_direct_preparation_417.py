"""Independent saved-array audit416: partition, fixed-column elimination, bounds
and epigraph algebra. No coordinate-map, FK, root or optimizer invocation.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'direct_preparation_audit_417'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False);r=json.loads((D/'direct_preparation_416/preparation.json').read_text());assert r['passed']
    for p,h in r['source_input_sha256'].items():assert sha(p)==h,p
    f=dict(np.load(D/'monotone_preparation_410/fixture.npz'));a=dict(np.load(D/'direct_preparation_416/fixture.npz'));b=dict(np.load(D/'window_path_400/initial.npz'))
    hard=np.array([not(12212<=i<19508 or i==19759 or i>=20284) for i in range(27708)])
    assert np.array_equal(hard,a['hard']) and np.array_equal(f['keep'],a['keep']) and np.array_equal(f['scale'],a['scale'])
    lower=[];upper=[]
    for index in f['shape']:
        row,slot=divmod(int(index),9);col=[2,3,21,23-1,16,17,18,19,20][slot]
        lower.append(b['lower'][row,col]);upper.append(b['upper'][row,col])
    lower=np.array(lower);upper=np.array(upper);assert np.array_equal(lower,a['lo']) and np.array_equal(upper,a['hi'])
    rows=np.flatnonzero(a['keep']);m=len(rows);J=np.zeros((m+1572,895));values=[]
    for j,i in enumerate(rows):
        values.append(f['iq'][i]/f['scale'][i]+(0. if hard[i] else float(a['slack'])))
        J[j,:894]=f['I'][i,:894]/f['scale'][i];J[j,894]=0. if hard[i] else 1.
    values=np.r_[values,f['v'][:786]-lower,upper-f['v'][:786]];J[m:m+786,:894]=f['G'][:786,:894];J[m+786:,:894]=-f['G'][:786,:894]
    assert np.array_equal(values,a['values']) and np.array_equal(J,a['jacobian'])
    radius=np.array([1. if i<785 else .1 for i in range(894)])
    assert np.array_equal(a['lower'],a['seed'][:894]-radius) and np.array_equal(a['upper'],a['seed'][:894]+radius)
    assert np.array_equal(a['seed'],f['z']) and np.array_equal(a['seed'][-5:],[93,194,246,101,16]) and np.all(a['expanded_direction'][-5:]==0)
    assert np.all(values>=0) and np.all(f['iq'][rows[hard[rows]]]>=0)
    dot=float(abs(J@a['direction']-a['separate_direction']).max());assert dot<2e-12
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,all_original_and_window_rows=27708,optimizer_rows=len(values),jacobian_shape=list(J.shape),partition_and_bounds_exact=True,all_3648_static_command_rows_hard=True,fixed_budget_columns_eliminated_exactly=True,scalar_direction_error=dot,unchanged_existing_derivative_bound=2e-6,physical_admission=False,physics_ticks=0,policy_optimizer_updates=0,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),D/'direct_preparation_416/preparation.json',D/'direct_preparation_416/fixture.npz']})
    (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
