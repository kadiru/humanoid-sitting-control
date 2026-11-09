"""Explicit binary64 surrogate for586: inward inequalities and certified equality defects."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,math,struct,shutil
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
SPEC=dict(
 scope='NEW numeric INPUT preparation for586 exact rational coupled-foot dimension reduction. No LP/primalpoint/graphlift/actualmechanics. Complete numerical algorithm,wholepath and return auditor remain absent.',
 source='Immutable586 rational A AND rational AFFINE E/RHS;587 inverse/graph/reverse proofs inherited byhash. No block inversion/substitution/old578 graph proof or580 encoding repeated. Original472actualX/Y/J,490canonicalaux anddrops unchanged.',
 domain='Project the ALREADY certified580 explicit OUTER coordinate box by586 retained_columns into the SAME original coordinates. Prove index association using580 retained_columns and586 original_columns,not an assumption about changed coefficients. No new endpoint derivation or power2 encoding. This is a new14636-coordinate domain projection,not reuse of580 row compensation. Original586 native DIRECTbounds remain infinite and are preserved separately. Impose every projected numeric bound explicitly;scalaroriginal78790 at14635 retains original[0,U]. Exact586 feasibility lifts to578 so inherited580 coordinate enclosures apply to the projected coordinates; approximate points do not inherit omitted-coordinate feasibility.',
 inequality='For every NEW586 rational inequality a.v<=b,encode each coefficient once nearest-evenfinitebinary64 ahat. Compute exact e=a-F(ahat),M=sum max(e*l,e*u) over projected explicit bounds. bhat=floor64(b-M) by direct integer IEEE encoding. Save M,b-M,downward rounding slack. All265967rows/1891613slots retained,even roundedzero slots;no scaling/threshold/rowremoval/adaptiveguard. ExactREAL numeric inequality satisfaction insidebox implies exactrational inequality satisfaction. No tolerance-equivalence.',
 equality='For EVERY5462 NEW586 rational AFFINE equality c.v=d,encode each coefficient chat and RHS dhat ONCE nearest-evenfinitebinary64. Define e=c-F(chat),delta=d-F(dhat),L=sum min(e*l,e*u)-delta,H=sum max(e*l,e*u)-delta. Save exact L,H,delta,rho=max(abs(L),abs(H)). For any REALv inside explicitbox, exactrational equality residual=(chat.v-dhat)+error with error in[L,H]. E remains an equality numerically,not replaced by inequalities orzeroRHS. All5462rows/52506slots retained,including1366zeroE/scalarzero. No claim that chat.v=dhat implies exact c.v=d.',
 equality_policy='The COMPLETE numeric system is an APPROXIMATE EQUALITY SURROGATE,NOT a certified REALinner/equivalent feasible set of586. Only its inequality block has the sufficient-inner implication. Equality defect intervals bound coefficient/RHS encoding error ONLY,not backend dot-product error or solver residual. No tolerance/guard change and no guaranteed feasibility/existence. Future return must independently evaluate EVERY exact rational586 equality at unchanged backendcoordinates and reject if exact residual exceeds unchanged1e-9;also all exact586 inequalities,fulloriginal checks andnumericKKT. Worst-case interval need not establish an unconditional residual budget;no adaptive reencoding or coefficientrepair based on certificate sizes.',
 finite='Future14636retainedvaluesBITWISEbackend,scalarALREADYexact0(+/-0),no postsetting. Exactnewfootgraphs thenold578auxgraphs on EXACTUNROUNDED intermediate31016-vector;roundeach64155eliminatedoriginalcoordinateONCE. No universalfiniteextension/no clip/repair/normalize. Fullnumeric588,exact586/578/572/570/546/516/FULL490 primal/equality/bound/requiredstationarity/absgap<=1e-9 inside1e-7.573analyticlower_t1 onlyinherited,no newdualproof/primalpair. NumericalKKTNOTexactfeasibility/optimality/objectiveerrorbound.',
 wholepath='Freeze complete initialization/algorithm/return and independentactualstateaudit BEFOREanyLP. SAMEcompleteoriginaldirectioncomputesmodeledphi/theta/ACTUAL(mu+theta)/2ceiling/prediction,primaryNEVERmapped/notassumedphi0. Actualinterior needs freshFULLoriginal490 conservativebranch;zero-facegraphs inapplicable. Newstate/trust freshassembly/validproofs beforeLPinside45sTOTALMAINentrybeforeimports/hash/preparation/backend/allKKT.40GiB/120sparent,24major48LP120maps5dyadics/trust/guards unchanged. Exactoriginalunscaledeta>0before490encoder,newXfreshmechanics/J. Allphysical/filter/runtime/72retention/PPOgates unchanged.',
 audit='Standalone589 importsNOproducer/optimizer. Verify ALLnewnearestcoefficient/RHS cells by exact adjacentbinary64midpoint/tie-even comparisons; independent integer-cross-difference products forA support andE lower/upperdefects; ALLA directedRHS successor floors; projectedbox originalcoordinate association andcost/maps/IDs. No oldencoding/inversion/graph/analyticdualproof replay.',
 resources='ONE240sproducer andONE240saudit TOTALbefore scientificimports/inputhashes,40GiBaddressspace,checksbetweenoperations,peakUNMEASURED. Preparationcaps NOT45srobotLPextension. Failureterminal,no completedarithmeticreplay.',
 physical_admission=False,complete_numerical_algorithm_exists=False,primal_witness_exists=False,
 complete_numeric_system_real_inner_subset_claimed=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def check(start):
 if time.monotonic()-start>240:raise TimeoutError('240s foot numeric-input preparation cap')
def pair(q):return [q.numerator,q.denominator]
def floor64(q):
 if not q:return 0.0
 sign=1 if q>0 else -1;n=abs(q.numerator);d=q.denominator;e=n.bit_length()-d.bit_length()
 if (n<d<<e) if e>=0 else (n<<-e<d):e-=1
 k=max(-1074,e-52);num,den=(n,d<<k) if k>=0 else (n<<-k,d);m,rem=divmod(num,den)
 if sign<0 and rem:m+=1
 if m==2**53:m//=2;k+=1
 if not m:bits=0
 elif k==-1074 and m<2**52:bits=m
 else:
  ex=k+52+1023;assert 1<=ex<=2046 and 2**52<=m<2**53,'finite RHS encoding range'
  bits=(ex<<52)|(m-2**52)
 if sign<0 and m:bits|=1<<63
 v=struct.unpack('>d',bits.to_bytes(8,'big'))[0];assert math.isfinite(v) and F(v)<=q;return v
def encode(src,out,domain,kind,start):
 m,nnz=(265967,1891613) if kind=='inequality' else (5462,52506)
 data=np.empty(nnz);indices=np.empty(nnz,dtype=np.int32);indptr=np.zeros(m+1,dtype=np.int32);rhs=np.empty(m)
 counts=dict(rows=m,coefficient_encodings=nnz,inexact_coefficients=0,rounded_zero_coefficients=0,error_endpoint_products=0,rhs_encodings=m,inexact_rhs=0,nonzero_error_rows=0,maximum_fraction_bits=0)
 maxradius=F(0);maxdown=F(0);slot=0;first_excess=None;excess=0
 source_name='rational_inequalities.jsonl' if kind=='inequality' else 'rational_equalities.jsonl'
 with (src/source_name).open() as f,(out/(kind+'_certificates.jsonl')).open('x') as log:
  for i,line in enumerate(f):
   if i%64==0:check(start)
   row,terms,b=json.loads(line);assert 0<=i<m
   # Source row identifiers are inherited578 positions; inequalities remain0..m-1.
   if kind=='inequality':assert row==i
   lo_sum=F(0);hi_sum=F(0)
   for col,n,d in terms:
    q=F(n,d);v=float(q);assert math.isfinite(v);err=q-F(v);lo,hi=domain[col]
    if kind=='inequality':hi_sum+=err*(hi if err>0 else lo);counts['error_endpoint_products']+=1
    else:
     lo_sum+=err*(lo if err>=0 else hi);hi_sum+=err*(hi if err>=0 else lo);counts['error_endpoint_products']+=2
    counts['inexact_coefficients']+=int(err!=0);counts['rounded_zero_coefficients']+=int(v==0)
    data[slot]=v;indices[slot]=col;slot+=1
   exact_rhs=F(*b)
   if kind=='inequality':
    threshold=exact_rhs-hi_sum;v=floor64(threshold);down=threshold-F(v);assert down>=0
    cert=[i,row,pair(hi_sum),pair(threshold),pair(down)];values=[hi_sum,threshold,down]
    counts['inexact_rhs']+=int(down!=0);counts['nonzero_error_rows']+=int(hi_sum!=0);maxradius=max(maxradius,abs(hi_sum));maxdown=max(maxdown,down)
   else:
    v=float(exact_rhs);assert math.isfinite(v);delta=exact_rhs-F(v);low=lo_sum-delta;high=hi_sum-delta;assert low<=high
    radius=max(abs(low),abs(high));cert=[i,row,pair(low),pair(high),pair(delta),pair(radius)];values=[low,high,delta,radius]
    counts['inexact_rhs']+=int(delta!=0);counts['nonzero_error_rows']+=int(radius!=0);maxradius=max(maxradius,radius)
    if radius>F(1e-9):
     excess+=1
     if first_excess is None:first_excess=i
   for q in values:counts['maximum_fraction_bits']=max(counts['maximum_fraction_bits'],abs(q.numerator).bit_length(),q.denominator.bit_length())
   rhs[i]=v;indptr[i+1]=slot;log.write(json.dumps(cert,separators=(',',':'))+'\n')
 assert i+1==m and slot==nnz;check(start)
 mat=sparse.csr_matrix((data,indices,indptr),shape=(m,14636));assert mat.has_canonical_format
 sparse.save_npz(out/(kind+'_matrix.npz'),mat)
 counts.update(maximum_exact_error_radius=str(maxradius),maximum_directed_rhs_slack=str(maxdown),equality_defect_radius_above_1e_9=excess,first_equality_radius_above_1e_9=first_excess)
 return rhs,counts
def build(out,start):
 src=D/'foot_block_preparation_586';old=dict(np.load(src/'arrays.npz'));maps=dict(np.load(src/'maps.npz'))
 b580=dict(np.load(D/'graph_encoding_preparation_580/arrays.npz'));m580=dict(np.load(D/'graph_encoding_preparation_580/maps.npz'))
 selected=maps['retained_columns'];original=maps['original_columns'];assert selected.shape==original.shape==(14636,)
 assert np.array_equal(m580['retained_columns'][selected],original) and original[-1]==78790 and selected[-1]==31015
 bounds=b580['bounds'][selected].copy();assert np.isfinite(bounds).all() and bounds.shape==(14636,2)
 assert np.isneginf(old['bounds'][:-1,0]).all() and np.isposinf(old['bounds'][:-1,1]).all()
 assert np.array_equal(bounds[-1].view(np.uint64),old['bounds'][-1].view(np.uint64))
 domain=[(F(float(lo)),F(float(hi))) for lo,hi in bounds];assert all(lo<=hi for lo,hi in domain)
 arhs,ac=encode(src,out,domain,'inequality',start);erhs,ec=encode(src,out,domain,'equality',start)
 np.savez_compressed(out/'arrays.npz',rhs=arhs,equality_rhs=erhs,cost=old['cost'],bounds=bounds)
 np.save(out/'original_bounds.npy',old['bounds'])
 for name in ['maps.npz','original_row_ids.npy']:shutil.copy2(src/name,out/name)
 check(start)
 return dict(columns=14636,original_scalar_column=78790,reduced_scalar_column=14635,projected_existing_box_coordinates=14636,new_endpoint_encodings=0,inequality=ac,equality=ec,original_inequalities_removed=0,original_equalities_removed=0,equalities_copied_binary64=False,complete_system_is_equality_surrogate=True,complete_system_real_inner_subset_claimed=False,inequality_inner_implication_only=True,graph_point_lifts=0,old_proofs_repeated=False)
