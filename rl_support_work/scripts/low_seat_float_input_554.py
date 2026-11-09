"""NEW554 finite-domain adapter; reference552 is inherited, never recomputed."""
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from scipy import sparse
import json,sys
from low_seat_box_input_548 import quotient,derive,sha,read,D
from low_seat_float_hull_552 import directed
def derive_hull(src,out,check):
 A=sparse.load_npz(src/'matrix.npz').tocsr();a=dict(np.load(src/'arrays.npz'));E=sparse.load_npz(src/'equality.npz');ep=read(src/'endpoints.json');owners=np.load(src/'bound_owners.npy');ids=np.load(src/'original_row_ids.npy');oldkeep=np.load(src/'retained_source_positions.npy')
 assert A.shape[1]==78791 and A.has_canonical_format and np.all(A.data!=0) and len(ep)==A.shape[1] and np.all(owners!=-2)
 assert sys.float_info.radix==2 and sys.float_info.mant_dig==53 and sys.float_info.max_exp==1024 and sys.float_info.min_exp==-1021 and np.dtype('float64').itemsize==8
 bounds=np.empty((A.shape[1],2));enc=np.empty((A.shape[1],2,3),dtype=np.int64);changed=0;worst=F(0);bitmax=0
 with (out/'rounding_ledger.jsonl').open('w') as log:
  for j,(ls,us) in enumerate(ep):
   check();lo,hi=F(ls),F(us);assert lo<=hi
   bounds[j,0],enc[j,0]=directed(lo,True);bounds[j,1],enc[j,1]=directed(hi,False)
   assert bounds[j,0]<=bounds[j,1],'empty binary64 interval: no preparation output/candidate accepted'
   dl=F(float(bounds[j,0]))-lo;du=hi-F(float(bounds[j,1]));assert dl>=0 and du>=0
   changed+=int(dl>0)+int(du>0);worst=max(worst,dl,du);bitmax=max(bitmax,dl.numerator.bit_length(),dl.denominator.bit_length(),du.numerator.bit_length(),du.denominator.bit_length())
   log.write(json.dumps([j,str(dl),str(du)],separators=(',',':'))+'\n')
 keep=np.flatnonzero(np.diff(A.indptr)!=1);removed=np.flatnonzero(np.diff(A.indptr)==1);B=A[keep].tocsr();check()
 sparse.save_npz(out/'matrix.npz',B);np.savez_compressed(out/'arrays.npz',**dict(a,rhs=a['rhs'][keep],bounds=bounds));np.save(out/'original_bounds.npy',a['bounds']);np.save(out/'encoding.npy',enc);np.save(out/'retained546_positions.npy',keep);np.save(out/'removed546_positions.npy',removed);np.save(out/'source516_positions.npy',oldkeep[keep]);np.save(out/'original490_row_ids.npy',ids[keep]);np.save(out/'bound_owners516.npy',owners)
 import shutil
 shutil.copy2(src/'equality.npz',out/'equality.npz')
 return dict(source_shape=list(A.shape),source_nonzeros=int(A.nnz),new_shape=list(B.shape),new_nonzeros=int(B.nnz),singleton_rows_absorbed=len(removed),constants_retained=int(np.count_nonzero(np.diff(A.indptr)==0)),mixed_rows_retained=int(np.count_nonzero(np.diff(A.indptr)>=2)),endpoint_encodings=2*A.shape[1],strict_inward_endpoint_changes=changed,exact_endpoints_unchanged=2*A.shape[1]-changed,maximum_exact_inward_shift=str(worst),largest_shift_fraction_bits=bitmax,native_variables_removed=0,scalar_variables_removed=0,finite_binary64_feasible_set_equal=True,real_polytope_inner_subset=True,real_polytope_equivalence_claimed=False,tolerance_feasible_set_equivalence_claimed=False,source_endpoint_proof_inherited=True,new_endpoint_construction=False,new_mixed_support_tests=0,new_matrix_coefficients=False,new_candidate=False)

def lift_duals(SA,sa,hull,ry,upper,lower,out,check):
 """Exact-rational original dual lift, then one binary64 rounding per multiplier.
 For selected a*x_j<=b, transfer mu to mu/abs(a). Signed coefficient
 equals +/-mu exactly before rounding. Direct-bound owners copy mu.
 The old dual objective is new dual objective minus
 delta=sum(muL*(l64-L)+muU*(U-u64))>=0. Thus original signed gap
 increases by delta even with an exact lift; new KKT alone is insufficient.
 Rounded multiplier effects are recorded and ALL original KKT rechecked.
 This changes dual algebra only; no primal coordinate is reconstructed.
 """
 owners=np.load(hull/'bound_owners516.npy');keep=np.load(hull/'source516_positions.npy');z=dict(np.load(hull/'arrays.npz'));sy=np.zeros(SA.shape[0]);sy[keep]=ry;ol=np.zeros(SA.shape[1]);ou=ol.copy();delta=F(0);stationerr=F(0);rhs_round=F(0);count=0;seen=set()
 assert np.all(np.isfinite(upper)) and np.all(np.isfinite(lower)) and np.all(upper>=0) and np.all(lower>=0)
 ep=read(Path(read(out/'input_proof.json')['box_reduced'])/'endpoints.json')
 with (out/'dual_lift.jsonl').open('w') as log:
  for j in range(SA.shape[1]):
   check()
   for side,mus in [(0,lower),(1,upper)]:
    mu=F(float(mus[j]));owner=int(owners[j,side]);q=F(ep[j][side]);new=F(float(z['bounds'][j,side]));shift=new-q if side==0 else q-new;assert shift>=0;delta+=mu*shift
    if owner<0:
     assert owner==-1 and q==F(float(sa['bounds'][j,side]));(ou if side else ol)[j]=mus[j];continue
    if not mu:continue
    k=SA.indptr[owner];assert SA.indptr[owner+1]-k==1 and SA.indices[k]==j and owner not in seen;seen.add(owner)
    a=F(float(SA.data[k]));b=F(float(sa['rhs'][owner]));assert (a>0)==bool(side) and a*q==b and sy[owner]==0
    ratio=mu/abs(a);v=float(ratio);assert np.isfinite(v) and v>=0;err=F(v)-ratio;sy[owner]=v;stationerr=max(stationerr,abs(a*err));rhs_round+=b*err;count+=1
    log.write(json.dumps([j,side,owner,str(ratio),str(err),str(a*err),str(b*err)],separators=(',',':'))+'\n')
 record=dict(nonzero_row_transfers=count,exact_endpoint_gap_penalty=str(delta),exact_endpoint_gap_penalty_float=float(delta),maximum_exact_stationarity_rounding_error=str(stationerr),exact_rhs_rounding_contribution=str(rhs_round),original_signed_gap_identity='gap_original_exact_lift=gap_new+endpoint_gap_penalty; rounded lift additionally changes rhs dot dual by recorded exact_rhs_rounding_contribution',primal_coordinates_modified=False,original_real_LP_equivalence_claimed=False)
 (out/'dual_lift.json').write_text(json.dumps(record,indent=2));check();return sy,ou,ol
