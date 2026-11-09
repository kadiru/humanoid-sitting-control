"""NEW map arithmetic preparation only; no FK, root, AD or trajectory solve."""
import json,datetime
from scipy import sparse
from low_seat_bounded_retraction_454 import *
O=D/'bounded_retraction_preparation_454'
def main():
 O.mkdir(exist_ok=False);d=load_saved();c=d['reference'];lo=d['lower'];hi=d['upper'];z=np.zeros(len(c))
 exact,_,_=bounded_controls(c,z,lo,hi);assert np.array_equal(exact,c)
 sign=np.where(hi-c>=c-lo,1.,-1.);room=np.where(sign>0,hi-c,c-lo)
 assert np.all(room>0);t=sign*room*.125;h=room*1e-4
 q,a,b=bounded_controls(c,t,lo,hi);qp,_,_=bounded_controls(c,t+h,lo,hi);qm,_,_=bounded_controls(c,t-h,lo,hi)
 fd=(qp-qm)/(2*h);analytic=np.exp(-abs(t)/room);err=float(abs(fd-analytic).max());assert err<2e-6,err
 # The new feasible tangent at zero is identity, so the audited saved full
 #constraint Jacobian and all partitions are reused without another model run.
 chain=sparse.load_npz(D/'clock_row_preparation_v2_446/global_jacobian.npz').multiply((1/d['a']['scale'])[:,None]).tocsr()
 np.savez_compressed(O/'map_fixture.npz',origin=c,lower=lo,upper=hi,tangent=t,h=h,mapped=q,alpha=a,endpoint=b,plus=qp,minus=qm,analytic=analytic,finite_difference=fd)
 np.savez_compressed(O/'zero_map.npz',mapped=exact,reference=c)
 sparse.save_npz(O/'saved_zero_chain.npz',chain)
 # Exact endpoint outward directions are deliberately rejected, not clamped.
 failures=[]
 for c0,d0 in [(0.,-1.),(1.,1.)]:
  try:bounded_controls(np.array([c0]),np.array([d0]),np.array([0.]),np.array([1.]))
  except ValueError as e:failures.append(str(e))
 assert len(failures)==2
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,specification=SPEC,free_controls=len(c),seed_reconstruction_exact=True,map_derivative_maximum_error=err,derivative_bound=2e-6,minimum_mapped_bound_margin=float(min((q-lo).min(),(hi-q).min())),endpoint_outward_rejections=failures,model_calls=0,root_calls=0,optimizer_calls=0,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
