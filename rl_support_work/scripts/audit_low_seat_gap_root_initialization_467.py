"""Independent saved-array audit of additional unpredicted-start root fixtures only."""
from pathlib import Path
import numpy as np,json,datetime,hashlib
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'gap_root_initialization_467';O=D/'gap_root_initialization_audit_467'
def main():
 assert json.loads((D/'gap_root_initialization_exit_467.json').read_text())['actual_exit']==0
 for n,h in json.loads((P/'SHA256.json').read_text()).items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h,n
 O.mkdir(exist_ok=False);a=dict(np.load(P/'minus.npz'));b=dict(np.load(P/'plus.npz'));f=dict(np.load(P/'comparison.npz'));v=dict(np.load(D/'gap_coordinate_preparation_v3_466/implicit_map.npz'));blocks=f['blocks'];h=float(f['h']);dep=[0,1,2]+list(range(4,16));ind=[3]+list(range(16,24));direction=f['direction'];hist=json.loads((P/'journal.json').read_text())
 for sign,z,H in [(-1,a,hist[0]),(1,b,hist[1])]:
  assert np.array_equal(z['initial'][:,dep],v['referenceX'][blocks][:,dep]);assert np.array_equal(z['X'][:,ind],z['initial'][:,ind]);assert np.array_equal(z['initial'][:,ind],v['referenceX'][blocks][:,ind]+sign*h*direction[:,1:]);assert len(H)>1 and H[-1]<=1e-13;assert abs(z['Y'][:,:14]).max()<=1e-13;assert abs(z['Y'][:,47]-z['target_gap']).max()<=1e-13
 fd=(b['X']-a['X'])/(2*h);predict=np.stack([v['T'][j]@direction[i] for i,j in enumerate(blocks)]);err=float(abs(fd-predict).max());assert err<2e-6
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,passed=True,derivative_error=err,unchanged_bound=2e-6,dependent_initial_states_exact_unperturbed=True,independent_controls_exact=True,both_roots_performed_updates=True,saved_newton_updates=[len(x)-1 for x in hist],new_model_or_root_calls=0,original_preparation_and_full_audit_not_repeated=True,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
