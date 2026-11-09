"""Saved-array independent identity/moment/projection/time audit; no FK or root."""
from pathlib import Path
from fractions import Fraction
import json,hashlib,datetime
import numpy as np
from scipy.spatial import ConvexHull
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'contact_lift_preparation_audit_439'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);read=lambda p:json.loads(p.read_text());r=read(D/'contact_lift_preparation_438/preparation.json');f=dict(np.load(D/'contact_lift_preparation_438/fixture.npz'));assert r['passed']
 for p,h in read(D/'contact_lift_preparation_438/SHA256.json').items():assert sha(D/'contact_lift_preparation_438'/p)==h
 errors=dict(identity=0.,moment=0.,normal=0.,projection_kkt=0.,derivative=0.,event=0.)
 W=np.zeros((115,8))
 for k in range(115):
  for slot in range(8):
   if f['old_weights'][k,slot]==0:assert f['mapping'][k,slot]==-1;continue
   dest=int(f['mapping'][k,slot]);assert dest//4==slot//4 and W[k,dest]==0;W[k,dest]=f['old_weights'][k,slot]
   j=slot//4;local=f['feet'][k,j,:3,:3].T@(f['old_points'][k,slot]-f['feet'][k,j,:3,3]);errors['identity']=max(errors['identity'],float(abs(local-f['local'][j,dest%4]).max()))
 assert np.array_equal(W,f['source_weights'])
 changed={v['knot']:v for v in r['changed_nominal_nodes']}
 for k in range(115):
  for j in range(2):
   sl=slice(j*4,(j+1)*4);errors['normal']=max(errors['normal'],float(abs(sum(f['weights'][k,sl])-sum(W[k,sl]))))
  if k not in changed:assert np.array_equal(f['weights'][k],W[k])
  else:
   item=changed[k];a=np.array(item['original_com_xy']);b=np.array(item['new_com_xy']);H=ConvexHull(f['points'][k,:4,:2]).equations;A=H[:,:2];rhs=-H[:,2]-r['projection_inset_m'];active=item['active_polygon_edges'];lam=np.linalg.lstsq(A[active].T,a-b,rcond=None)[0];assert lam.min()>=0
   errors['projection_kkt']=max(errors['projection_kkt'],float(abs(A[active].T@lam-(a-b)).max()),float(max(0.,-(rhs-A@b).min())))
   errors['moment']=max(errors['moment'],float(abs(f['weights'][k]@f['points'][k,:,:2]-b).max()))
  assert np.all(f['weights'][k]>=0)
 assert np.all(f['weights'][16:65,4:]==0)
 times=[Fraction(int(a),int(b)) for a,b in r['vertex_times']];assert all(a<b for a,b in zip(times[:-1],times[1:]));assert times[-1]==Fraction(27,2)
 for node,total in [(16,93),(64,287),(99,532),(105,658),(115,675)]:assert times[node]==Fraction(total,50),(node,times[node])
 # The derivative fixture contains independent central differences evaluated in
 # preparation; this audit verifies every stored column comparison, not new FK.
 for i,item in enumerate(r['root_derivative_fixtures']):
  error=float(abs(f[f'J_{i}']-f[f'FD_{i}']).max());assert error==item['error'] and error<2e-6;errors['derivative']=max(errors['derivative'],error)
 assert max(errors[x] for x in ['identity','normal','moment','projection_kkt'])<2e-10
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,nodes=115,unchanged_load_nodes=115-len(changed),new_nominal_nodes=sorted(changed),maximum_errors=errors,root_projection_calls=0,finite_difference_columns_repeated=0,actual_forward_filter_ticks=0,physics_ticks=0,complete_path_or_physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
