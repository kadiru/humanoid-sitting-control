"""Independent saved-map rational containment/derivative/chain audit. No FK."""
from pathlib import Path
from fractions import Fraction
import json,datetime,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'bounded_retraction_preparation_454';O=D/'bounded_retraction_audit_455'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);f=dict(np.load(P/'map_fixture.npz'));z=np.load(P/'zero_map.npz');assert np.array_equal(z['mapped'],z['reference']);n=len(f['origin']);rebuilt=[]
 for i in range(n):
  c,a,b=map(Fraction.from_float,[float(f[k][i]) for k in ['origin','alpha','endpoint']]);q=c+a*(b-c)
  assert min(c,b)<=q<=max(c,b);v=float(q);assert float(f['lower'][i])<=v<=float(f['upper'][i]);rebuilt.append(v)
 assert np.array_equal(rebuilt,f['mapped']);room=abs(f['endpoint']-f['origin']);analytic=np.exp(-abs(f['tangent'])/room)
 fd=(f['plus']-f['minus'])/(2*f['h']);assert np.array_equal(fd,f['finite_difference']);err=float(abs(fd-analytic).max());assert err<2e-6
 V=D/'clock_row_preparation_v2_446';r=np.load(V/'row_arrays.npz');A=sparse.load_npz(V/'global_jacobian.npz').multiply((1/r['scale'])[:,None]);diff=A-sparse.load_npz(P/'saved_zero_chain.npz');ce=0. if not diff.nnz else float(abs(diff.data).max());assert ce==0
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,all_coordinate_rational_replay_exact=True,coordinates=n,feasible_zero_tangent_chain_error=ce,map_derivative_error=err,derivative_bound=2e-6,root_calls=0,model_calls=0,optimizer_calls=0,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
