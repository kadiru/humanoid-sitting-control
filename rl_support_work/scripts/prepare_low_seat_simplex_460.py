"""New saved-array chart, null-space chain and rounding-bound preparation.
No FK, root, AD, trajectory optimizer, physical simulation or old fixture rerun.
"""
import json,datetime
from low_seat_simplex_retraction_460 import *
O=D/'simplex_preparation_460'
def main():
 O.mkdir(exist_ok=False);d=load_saved();groups,R=layout(d);c=d['reference'];lo=d['lower'];hi=d['upper'];n=R.shape[1]
 zero,_,_=decode(c,np.zeros(n),lo,hi,groups);assert np.array_equal(zero,c)
 # Every independent coordinate gets its own NEW ray fixture. The derivative
 # along the nonzero ray is R[:,j]/(1+rho)^2. These are arithmetic derivatives,
 # not a second mechanics/kinematics derivative experiment.
 rows=[];fd_error=0.;joint=np.zeros(n)
 for gi,g in enumerate(groups):
  ids=g['indices'];cg=c[ids];lg=lo[ids];ug=hi[ids]
  for local,j in enumerate(g['columns']):
   direction=np.zeros(len(g['columns']));direction[local]=1.
   ds=np.asarray([float(v) for v in exact_direction(direction,g['pivot'],len(ids))])
   rooms=np.where(ds>0,ug-cg,cg-lg);room=min(rooms[ds!=0]/abs(ds[ds!=0]));t=.125*room;h=.001*t
   mid,alpha,_,_=decode_group(cg,direction*t,lg,ug,g['pivot'])
   plus,_,_,_=decode_group(cg,direction*(t+h),lg,ug,g['pivot']);minus,_,_,_=decode_group(cg,direction*(t-h),lg,ug,g['pivot'])
   fd=(plus-minus)/(2*h);analytic=ds*float(alpha)**2;err=float(abs(fd-analytic).max());fd_error=max(fd_error,err)
   rows.append(dict(group=gi,column=j,amplitude=t,h=h,error=err,alpha=float(alpha),plus=plus.tolist(),minus=minus.tolist(),middle=mid.tolist(),analytic=analytic.tolist()))
   joint[j]=t*(1.+.1*local)
 assert fd_error<CONTRACT['derivative_bound'],fd_error
 # A simultaneous all-coordinate fixture tests common load scaling, including
 # nonzero dependent pivots; no componentwise rescaling or normalization.
 mapped,alphas,sum_errors=decode(c,joint,lo,hi,groups)
 np.savez_compressed(O/'joint_fixture.npz',origin=c,lower=lo,upper=hi,tangent=joint,mapped=mapped,alphas=alphas,sum_errors=sum_errors,zero=zero)
 (O/'layout.json').write_text(json.dumps(groups));(O/'ray_fixtures.json').write_text(json.dumps(rows))
 sparse.save_npz(O/'null_basis.npz',R)
 V=D/'clock_row_preparation_v2_446';Q=D/'clock_row_preparation_446'
 A=sparse.load_npz(V/'global_jacobian.npz').multiply((1/d['a']['scale'])[:,None]).tocsr();E=sparse.load_npz(Q/'equality_jacobian.npz').tocsr()
 chain=(A@R).tocsr();eqchain=(E@R).tocsr();normal=eqchain[:-1].copy();normal.eliminate_zeros()
 assert normal.nnz==0,'normal tangent must be exactly zero, not thresholded'
 sparse.save_npz(O/'all_rows_null_chain.npz',chain);sparse.save_npz(O/'equalities_null_chain.npz',eqchain)
 # Loads lie in[0,1]. Their largest spacing strictly below1 is2^-53, hence
 # one correctly-rounded value errs by at most2^-54 (1 itself is exact).
 # Initial free-state seat loads are zero; their exact normal target is1.
 load_groups=[g for g in groups if g['kind']=='loads'];max_initial=0.;max_count=0
 source_normal=np.load(D/'contact_lift_440/states.npz')['normal'][d['g']['canonical']]
 assert np.all(source_normal[~d['g']['fixed']]==0)
 assert np.all(hi<=1.) and np.all(lo>=0.)
 for g in load_groups:
  w=d['initialX'][g['state'],16:24];exact=sum((F.from_float(float(x)) for x in w),F(0));max_initial=max(max_initial,float(abs(exact-1)));max_count=max(max_count,len(g['indices']))
 N=CONTRACT['max_consecutive_map_roundings'];u=CONTRACT['binary64_unit_roundoff'];round_bound=N*max_count*CONTRACT['max_round_error_per_load'];gamma7=7*u/(1-7*u)
 total_bound=max_initial+round_bound+gamma7*(1+max_initial+round_bound)
 assert total_bound<CONTRACT['original_total_normal_bound']
 assert np.max(abs(sum_errors))<=max_count*CONTRACT['max_round_error_per_load']
 proof=dict(max_initial_exact_load_sum_error=max_initial,maximum_active_loads=max_count,maximum_map_roundings=N,accumulated_exact_sum_rounding_bound=round_bound,sequential_sum_gamma7=gamma7,maximum_evaluated_normal_bound=total_bound,unchanged_normal_bound=1e-13,free_state_seat_load_exact_zero=True,scope='Free states have zero seat load; fixed states unchanged. Exact rational conservation precedes one rounding per load. This bound is not an actual-force certificate.')
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,contract=CONTRACT,free_controls=len(c),reduced_coordinates=n,load_groups=len(load_groups),groups=len(groups),original_inequalities=len(d['a']['scale']),original_equalities=E.shape[0],retained_rows=int(d['a']['keep'].sum()),retained_equalities=int(d['ea']['optimizer_keep'].sum()),normal_tangent_nonzeros=normal.nnz,ray_derivative_error=fd_error,unchanged_derivative_bound=2e-6,joint_bounds_minimum=float(min((mapped-lo).min(),(hi-mapped).min())),joint_maximum_exact_sum_change=float(abs(sum_errors).max()),rounding_proof=proof,all_old_rows_retained=True,seed_exact=True,model_calls=0,root_calls=0,optimizer_calls=0,candidate_exists=False,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
