"""Certified binary64 inner input for the new594 exact moment/gap condensation."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,math,struct,shutil
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
SPEC=dict(
 scope='NEW numeric INPUT preparation for594 exact rational moment/gap dimension reduction. No LP/primalpoint/graphpointlift/actualmechanics. Complete numericalalgorithm/wholepath/returnauditor remain absent.',
 source='Immutable594 exactA/E/RHS.595 exactinverse/graph/reverse proofs inheritedbyhash. No594/586 inversions/substitutions/reverseproofs,588 encodingcertificates,578 graphs,573analyticdual orothercompletedscience repeated. Actual472INITIALX/Y/J+490canonicalaux/drops unchanged.',
 domain='Project ALREADYcertified588 explicit finiteOUTERbounds using594 retained_columns. NEWassociation verifies588original_columns[selected]==594original_columns.10541coordinates=10540priorcontrols+scalaroriginal78790at10540. No newendpoint/power2 encoding oroldboxproof replay. Retained originalDIRECTbounds remaininfinite exceptscalar andsavedseparately. EVERYprojectedbound imposedexplicitly withnewcompensatedrows. Exact594feasibility lifts to586 so inheritedoriginalcoordinate enclosures apply; numericalapproximation doesNOTestablish feasibility ofeliminatedcoordinates.',
 inequality='Every NEW594rational a.v<=b: encode ahat once nearest-evenfinitebinary64, e=a-F(ahat),M=sum max(e*l,e*u) overexplicitprojectedbox, bhat=greatestfinitebinary64<=b-M usingdirectintegerIEEEencoding. Save exactM,b-M,downwardslack. PreserveALL265967rows/order/6102946slots INCLUDINGany roundedzero slots. No scaling/threshold/rowdeletion/normalization/adaptiveguard orcertificate-dependent retry. ExactREALnumericAsatisfactioninsidebox implies exactrationalAsatisfaction.',
 equality='Every1367rationalErow/RHS encodedonce andrequiredEXACTLYrepresentable:1366emptyzeroequalities pluslastscalar coefficientONEat10540,RHSzero. ALL1367rowsretained. Assert Fraction(binary64coefficient)==rationalcoefficient andFraction(binary64RHS)==rationalRHS withfullsourceIDs/positions. Unlike588 transformedmoment/gapE,594EcontainsnoinexactAFFINEcoefficient/RHS. Anyunexpectedlayout/nonexactencoding terminal,not anapproximateEsubstitute or adaptivepolicy. No endpoint-errorproductsneededforEXACTequalityencodings.',
 theorem='Inside theexplicitprojectedbox, numericREALAfeasibility implies rationalAfeasibility bysupportcompensation; numericEisEXACTrationalE. Therefore COMPLETE596numericREALfeasibleset is a SUFFICIENTINNERsubset of594 exactreducedzeroFACE,possiblyempty/proper. NOTfeasible-setequivalence,finite/toleranceequivalence,primalexistence/optimumzero/performance. Encodingerrorsupport/downwardslack areINPUTcertificates,notpointresiduals. Backenddot/solverresiduals andfinitecompositelift remainunchecked.',
 finite='Future10541backendretainedvaluesBITWISE,scalarALREADYexactzero(+/-0),no postsetting/repair/clip/normalize. Evaluate594rootxy/f graphs EXACTLY,586legs onEXACTUNROUNDED14636intermediate,578aux onEXACTUNROUNDED31016intermediate;round EACH68250eliminatedoriginalvalueONCE. No universalbinary64extension. Fullnumeric596/exact594/586/578/572/570/546/516/FULL490 primal/equality/bounds/requiredstationarity/absgap<=1e-9inside1e-7. Fullpoint/originalchecks notinferredfrominnerimplication/graphs.573analyticlower_t1dual inheritedonly,no newproof/pair;KKTnotexactoptimality/objectiveerrorbound.',
 wholepath='Completeinitialization/algorithm/return/independentactualstateaudit frozenBEFOREanyLP. SAMEcompleteoriginalprimarydirection computesmodeledphi/theta/ACTUAL(mu+theta)/2ceiling/prediction,primaryNEVERmapped/phiNOTassumed0. ActualinteriorfreshFULLoriginal490conservativebranch;zeroFACEgraphsinvalid. Changedstate/trustfreshfullassembly/validproofs orconservativeFULLoriginalbeforeLPwithin45sTOTALMAINentrybeforeimports/globalhash/prep/backend/point/lift/allKKT.40GiB/120sparent,24major48LP120maps5dyadics/trust/guardsunchanged. Exactunscaledeta>0before490encoder,freshnewXmechanics/J,no15rootdrift;allphysical/filter/runtime/72retention/PPOgates unchanged.',
 audit='Standalone597 importsNOproducer/optimizer. IndependentlyverifyALL6102946NEWnearestcoefficientcells byexactadjacentbinary64midpoint/tieeven,integercross-difference errorsupportproducts,ALL265967directedRHSsuccessorfloors,ALL1367exactErows/RHS andonecoefficient,projectedboxoriginalcoordinateassociation/cost/maps/IDs/originalbounds. Nooldencoding/block/graph/dualproof replay.',
 resources='ONE240sproducer andONE240saudit startingBEFOREscientificimports/globalhashes;40GiBAS/peakunmeasured. Checksbetweenrowsandhashchunks/serialization/outputwrites. Completiontiming sampledAFTERmainoutputhashes/manifest,excludesfinaltinycompletionrecord/stdout/processshutdown;doNOTclaimfullyinstrumentedprocesswall. Firstfailure/cap terminal,partialprefixpreservednotcertificate,no restart/completedarithmeticreplay. Symboliccaps NOT45srobotLPextension.',
 physical_admission=False,complete_numerical_algorithm_exists=False,primal_witness_exists=False,complete_numeric_system_real_inner_subset_claimed=True)
def check(start):
 if time.monotonic()-start>240:raise TimeoutError('240s balance numeric-input preparation cap')
def sha(p,start=None):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):
   if start is not None:check(start)
   h.update(b)
 return h.hexdigest()
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

def build(out,start,progress):
 src=D/'balance_block_preparation_594';old=dict(np.load(src/'arrays.npz'));maps=dict(np.load(src/'maps.npz'))
 existing=dict(np.load(D/'foot_encoding_preparation_588/arrays.npz'));em=dict(np.load(D/'foot_encoding_preparation_588/maps.npz'))
 selected=maps['retained_columns'];original=maps['original_columns'];assert selected.shape==original.shape==(10541,)
 assert np.array_equal(em['original_columns'][selected],original) and original[-1]==78790 and selected[-1]==14635
 bounds=existing['bounds'][selected].copy();assert np.isfinite(bounds).all() and bounds.shape==(10541,2)
 assert np.isneginf(old['bounds'][:-1,0]).all() and np.isposinf(old['bounds'][:-1,1]).all()
 assert np.array_equal(bounds[-1].view(np.uint64),old['bounds'][-1].view(np.uint64))
 domain=[(F(float(lo)),F(float(hi))) for lo,hi in bounds];assert all(lo<=hi for lo,hi in domain)
 m=265967;nnz=6102946;data=np.empty(nnz);indices=np.empty(nnz,dtype=np.int32);indptr=np.zeros(m+1,dtype=np.int32);rhs=np.empty(m)
 counts=dict(rows=m,coefficient_encodings=nnz,inexact_coefficients=0,rounded_zero_coefficients=0,error_endpoint_products=0,rhs_floor_encodings=m,strict_rhs_floors=0,nonzero_error_rows=0,maximum_fraction_bits=0)
 progress.update(stage='inequalities',completed_coefficient_encodings=0,completed_rhs_floors=0,completed_equality_rows=0,statistics=counts)
 maxsupport=F(0);maxdown=F(0);slot=0
 with (src/'rational_inequalities.jsonl').open() as f,(out/'inequality_certificates.jsonl').open('x') as log:
  for i,line in enumerate(f):
   if i%32==0:check(start)
   row,terms,b=json.loads(line);assert row==i and i<m;support=F(0)
   for col,n,d in terms:
    q=F(n,d);v=float(q);assert math.isfinite(v);err=q-F(v);lo,hi=domain[col]
    support+=err*(hi if err>0 else lo);counts['error_endpoint_products']+=1
    counts['inexact_coefficients']+=int(err!=0);counts['rounded_zero_coefficients']+=int(v==0)
    data[slot]=v;indices[slot]=col;slot+=1;progress['completed_coefficient_encodings']=slot
   threshold=F(*b)-support;v=floor64(threshold);down=threshold-F(v);assert down>=0
   counts['strict_rhs_floors']+=int(down!=0);counts['nonzero_error_rows']+=int(support!=0);maxsupport=max(maxsupport,abs(support));maxdown=max(maxdown,down)
   for q in [support,threshold,down]:counts['maximum_fraction_bits']=max(counts['maximum_fraction_bits'],abs(q.numerator).bit_length(),q.denominator.bit_length())
   rhs[i]=v;indptr[i+1]=slot;log.write(json.dumps([i,row,pair(support),pair(threshold),pair(down)],separators=(',',':'))+'\n');progress['completed_rhs_floors']=i+1
 assert i+1==m and slot==nnz;check(start)
 A=sparse.csr_matrix((data,indices,indptr),shape=(m,10541));assert A.has_canonical_format;sparse.save_npz(out/'inequality_matrix.npz',A);check(start)
 counts.update(maximum_exact_error_support=str(maxsupport),maximum_directed_rhs_slack=str(maxdown))
 progress['stage']='exact_equalities';erhs=np.zeros(1367);eptr=np.zeros(1368,dtype=np.int32)
 with (src/'rational_equalities.jsonl').open() as f,(out/'equality_certificates.jsonl').open('x') as log:
  for i,line in enumerate(f):
   if i%32==0:check(start)
   label,terms,h=json.loads(line);assert i<1367 and label==int(maps['source578_equality_rows'][i])
   assert terms==([] if i<1366 else [[10540,1,1]]) and h==[0,1]
   for j,n,d in terms:
    assert F(float(F(n,d)))==F(n,d);eptr[i+1]=eptr[i]+1
   v=float(F(*h));assert F(v)==F(*h);erhs[i]=v
   log.write(json.dumps([i,label,len(terms),True],separators=(',',':'))+'\n');progress['completed_equality_rows']=i+1
 assert i+1==1367 and eptr[-1]==1
 E=sparse.csr_matrix((np.array([1.0]),np.array([10540],dtype=np.int32),eptr),shape=(1367,10541));assert E.has_canonical_format
 sparse.save_npz(out/'equality_matrix.npz',E);check(start)
 np.savez_compressed(out/'arrays.npz',rhs=rhs,equality_rhs=erhs,cost=old['cost'],bounds=bounds);np.save(out/'original_bounds.npy',old['bounds'])
 for name in ['maps.npz','original_row_ids.npy']:shutil.copy2(src/name,out/name)
 check(start);progress['stage']='complete'
 return dict(columns=10541,original_scalar_column=78790,reduced_scalar_column=10540,projected_existing_box_coordinates=10541,new_endpoint_encodings=0,inequality=counts,equality=dict(rows=1367,coefficient_encodings=1,rhs_encodings=1367,inexact_coefficients=0,inexact_rhs=0,error_endpoint_products=0,coefficient_rhs_encoding_error_exactly_zero=True),original_inequalities_removed=0,original_equalities_removed=0,equalities_exact_binary64=True,complete_system_is_equality_surrogate=False,complete_system_real_inner_subset_claimed=True,graph_point_lifts=0,old_proofs_repeated=False)
