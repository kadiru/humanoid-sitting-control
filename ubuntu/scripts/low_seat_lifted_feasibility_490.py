"""Finite lifted mechanics-feasibility formulation; no executable run loop.

Affine output identities eliminate auxiliary feet/moments/normal. Actual X is
never projected by a15-root. Nonzero mechanics consistency means an infeasible
research iterate, not a path. All actual original criteria remain final gates.
"""
from fractions import Fraction as F
import numpy as np
from scipy import sparse
from low_seat_feature_collocation_488 import D,read,sha,load_saved,LOCAL_CONTROLS
from low_seat_exact_gap_chart_468 import load_domain,joint_round,gaps,fq,pair,unpack,U

OUTPUT=np.r_[np.arange(14),np.arange(15,51)]
EXTRA=np.r_[np.array([0,1,2]+list(range(4,16))),np.arange(39,71),np.arange(72,75)]
SPEC=dict(
 scope='NEW joint full-space minimax mechanics/rate feasibility formulation. Preparation/assembly/finite arithmetic only; no optimizer or full-path run wrapper.',
 seed='Sealed472 INITIAL X and actualY/J; new auxiliary identities explicitly canonicalized. No484 final/rejected state is used.',
 identities='Nonfixed auxiliary feet/moments=0; auxiliary normal=EXACT Fraction sum of binary64 loads minus1 rounded once; auxiliary gap=encoded target gap. Fixed45states untouched.',
 coordinates='10540 prior g/p/load/drop directions plus50 direct X/auxiliary coordinates per1365nonfixed state=78790. Analytic elimination of15 auxiliary identities from488, not deletion of actual criteria.',
 actual_consistency='50 outputs pernonfixed state: feet/moments14,commands16,torques16,gap1,COM3. All must match actual mechanics for admission. Actual normal checked separately at unchanged1e-13; normal consistency is a roundoff identity, not a zero residual restoration problem.',
 scaling='Frozen dimensional feature scales v=.02*sum(abs(original446G),axis=controls). Closure scale sigma=sum(abs(saved472J)*vX)+vY. This is numerical nondimensionalization, not a new physical threshold or new-state Jacobian.',
 merit='phi=max(0,negative normalized auxiliary rate/window rows,abs(50-output consistency/sigma)). Actual all331888 rows are separately mandatory and never replaced by auxiliary feasibility.',
 primary='Minimize one nonnegative feasibility epigraph for rate/window rows AND both signs of linearized50-output consistency. All156647 original affine HARD rows remain without epigraph. Nonzero consistency epigraph is explicitly INFEASIBLE mechanics, never admission.',
 witness='Second LP maximizes exact-certifiable relative interval clearance while retaining half the ACTUAL primary predicted phi reduction. Primary vector never decoded. This extends whole-feature consistency feasibility, not another482-only clearance run.',
 finite='One shared affine direction, exact Fraction row products, one binary64 rounding of direct features and loads. Existing468 directed joint drop encoding, exact pivot/endpoint sum, once-rounded cumulative gaps. No clipping/normalization/root fallback.',
 gap_reference='Canonical auxiliary gaps are explicitly assigned the saved decoded targets; inherited actual gaps are not silently treated as identical. Their initial consistency defects are saved. Normal-gap interval clearance reserves207U for old-gap rounding plus205U drop encoding plus new-gap rounding.',
 intervals='All11906 ORIGINAL full f/p/load boxes and206drop bounds, including pivots, plus per-feature trust intervals. Direct scalar endpoints one nextafter inward; normal-gap trust intervals207U inward; drop intervals U/205U inward. Exact positive certificate before every map; alpha>1 forbidden.',
 original_equalities='All1411 actual normal/hover primal tests retained. Under this EXPLICIT new identity elimination their tangent is zero; fixed-source constants and actual residuals remain checked. Do not apply this zero assembly to the uneliminated488 coordinates.',
 prospective_algorithm='At each major iteration recompute actual mechanics/J at current X, assemble primary and certified interior witness. Test at most five shared dyadic scales1,1/2,1/4,1/8,1/16, recertifying each unchanged scaled witness. No15-root. Post-map auxiliary hard rows/fullboxes/fixed/inactive must pass exactly. Recompute actualY; accept only positive phi reduction and actual/predicted>=.1 using the SAME encoded finite feature delta. An accepted off-consistency iterate is NOT a hard-feasible physical/model path. All actual rows/closure/geometry remain final admission gates.',
 prospective_budget=dict(max_major_iterations=24,max_actual_lps=48,max_finite_maps=120,lp_seconds=45,maximum_dyadic_trials=5,initial_trust=1.,minimum_trust=1/32,load_rounding_chain_cap=120),
 budget_rationale='One distinct24-major full-space minimax process, at most2LPs and5finite trials each;120maps preserves the existing load-rounding proof. No old run is extended. Full execution source and auditor still must be frozen before launch.',
 trust_policy='Existing trust1..1/32: halve after all5 trials reject; double up to1 only after acceptance ratio>.75. Stop belowfloor or atmajorcap; no polishing. Original440 +/- .02 global fullcontrol box unchanged.',
 numerical_limits='HiGHS1e-9/independentlinear1e-7; old2^-20 row guards; all8eps/4ULP/32ULP/2^-23 reserves unchanged. Max merit is a numerical feasibility objective, not physical severity or effort.',
 physical_admission=False,executable_run_wrapper=False)

def load():
 d=load_saved();s=d['seed'];d['dom']=load_domain();d['drops']=list(map(unpack,read(D/'free_gap_path_472/linear_00/metadata.json')['drops']))
 d['active']=np.flatnonzero(~d['mesh']['fixed']);K=len(s['X']);B=d['B'].tocoo();rr=(B.row//10)*75+LOCAL_CONTROLS[B.row%10];er=(d['active'][:,None]*75+EXTRA).ravel();n0=d['B'].shape[1]
 seat=np.load(D/'contact_lift_440/states.npz')['normal'][d['mesh']['canonical']]
 assert np.all(seat[d['active']]==0), 'Canonical normal identity requires zero nonfixed seat load'
 P=sparse.csr_matrix((np.r_[B.data,np.ones(len(er))],(np.r_[rr,er],np.r_[B.col,n0+np.arange(len(er))])),shape=(K*75,n0+len(er)))
 d['P']=P;d['extra_rows']=er;d['varying']=np.flatnonzero(np.diff(P.indptr));d['feature_scale']=.02*np.sum(abs(d['cm']['G']),axis=2)
 d['sigma']=np.einsum('nyx,nx->ny',abs(s['J']),d['feature_scale'][:,:24])+d['feature_scale'][:,24:]
 assert np.all(d['sigma'][d['active']][:,OUTPUT]>0) and np.all(d['feature_scale'].ravel()[d['varying']]>0)
 free=d['cm']['free'];ref=d['cm']['reference'].ravel()[free];d['lo']=np.maximum(d['cm']['lower'].ravel()[free],ref-.02);d['hi']=np.minimum(d['cm']['upper'].ravel()[free],ref+.02)
 d['control_feature_rows']=(free//10)*75+np.array([2,3]+list(range(16,24)))[free%10]
 nr=[];nc=[];nv=[]
 for j,c in enumerate(d['gap_layout']['columns']):
  if c['kind']=='normal_drop':nr.extend([c['edge'],c['pivot']]);nc.extend([j,j]);nv.extend([1.,-1.])
 d['N']=sparse.csr_matrix((nv,(nr,nc)),shape=(206,P.shape[1]));d['normal_states']=np.asarray(d['gap_layout']['normal_states']);d['normal_gap_rows']=set((d['normal_states']*75+71).tolist())
 d['source_actual']=np.c_[s['X'],s['Y']];d['initial']=canonical(d,d['source_actual'],s['target'][:,0]);assert P.shape==(105750,78790)
 return d

def canonical(d,features,target_gap=None):
 a=features.copy();active=d['active'];a[active,24:38]=0.
 for b in active:a[b,38]=float(sum(map(fq,a[b,16:24]),F(0))-1)
 if target_gap is not None:a[active,71]=target_gap[active]
 return a

def blocks(d,features,actualY,J):
 P=d['P'];a=d['rows'];r=(d['M']@features.ravel()+a['constant'])/a['scale'];eq=d['E']@features.ravel()+d['eq']['constant'];A=(d['M']@P).multiply((1/a['scale'])[:,None]).tocsr()
 active=d['active'];L=np.zeros((len(active),50,75));L[:,:,:24]=-J[active][:,OUTPUT,:]
 for i,y in enumerate(OUTPUT):L[:,i,24+y]=1.
 rows=(active[:,None]*75+np.arange(75)).ravel();C=(sparse.block_diag(list(L),format='csr')@P[rows]).multiply((1/d['sigma'][active][:,OUTPUT].ravel())[:,None]).tocsr()
 h=((features[active,24:][:,OUTPUT]-actualY[active][:,OUTPUT])/d['sigma'][active][:,OUTPUT]).ravel()
 soft=a['soft']&a['keep'];hard=a['keep']&~soft;variable=np.diff(A.indptr)>0;guard=np.zeros(len(r));guard[soft&variable]=2.**-20;hp=hard&variable;guard[hp]=np.where(r[hp]>0,np.minimum(2.**-20,.5*r[hp]),2.**-20)
 return dict(A=A,C=C,r=r,e=eq,h=h,guard=guard,phi=max(0.,float(-r[soft].min()),float(abs(h).max())))

def intervals(d,features,drops,trust):
 P=d['P'];ids=d['varying'];f=features.ravel();v=d['feature_scale'].ravel();H=sparse.vstack([P[d['control_feature_rows']],P[ids],d['N']],format='csr');lower=[];upper=[];cl=[];cu=[];owners=[]
 def add(l,u,c,owner,encoding=None):
  l=float(l);u=float(u);cl.append(fq(l)-c);cu.append(fq(u)-c)
  if encoding is None:lower.append(fq(np.nextafter(l,np.inf))-c);upper.append(fq(np.nextafter(u,-np.inf))-c)
  else:lower.append(fq(l)-c+encoding);upper.append(fq(u)-c-encoding)
  owners.append(owner)
 for i,row in enumerate(d['control_feature_rows']):add(d['lo'][i],d['hi'][i],fq(f[row]),['original_control',i])
 for row in ids:add(f[row]-trust*v[row],f[row]+trust*v[row],fq(f[row]),['feature_trust',int(row)],207*U if row in d['normal_gap_rows'] else None)
 for i,(c,l,u) in enumerate(zip(drops,d['dom']['lower'],d['dom']['upper'])):
  e=205*U if i==d['dom']['pivot'] else U;cl.append(l-c);cu.append(u-c);lower.append(l-c+e);upper.append(u-c-e);owners.append(['drop',i])
 widths=[u-l for l,u in zip(lower,upper)];assert all(w>0 for w in widths)
 return dict(H=H,lower=lower,upper=upper,widths=widths,closed_lower=cl,closed_upper=cu,owners=owners)

def assemble(d,b,s,ceiling=None):
 n=d['P'].shape[1];keep=d['rows']['keep'];soft=d['rows']['soft'][keep].astype(float);k=int(keep.sum());m=b['C'].shape[0];H=s['H'];width=np.array([float(x) for x in s['widths']]);HN=H.multiply((1/width)[:,None]).tocsr();closed_rhs=np.r_[[float(u/w) for u,w in zip(s['closed_upper'],s['widths'])],[-float(l/w) for l,w in zip(s['closed_lower'],s['widths'])]]
 if ceiling is None:
  top=sparse.hstack([-b['A'][keep],sparse.csr_matrix(-soft[:,None])]);cr=sparse.csr_matrix(-np.ones((m,1)));closure=sparse.vstack([sparse.hstack([b['C'],cr]),sparse.hstack([-b['C'],cr])]);rhs=np.r_[b['r'][keep]-b['guard'][keep],-b['h'],b['h'],closed_rhs];cost=np.r_[np.zeros(n),1.];last=[0.,max(1.,2*b['phi'])]
 else:
  top=sparse.hstack([-b['A'][keep],sparse.csr_matrix((k,1))]);closure=sparse.vstack([sparse.hstack([b['C'],sparse.csr_matrix((m,1))]),sparse.hstack([-b['C'],sparse.csr_matrix((m,1))])]);rhs=np.r_[b['r'][keep]-b['guard'][keep]+soft*ceiling,-b['h']+ceiling,b['h']+ceiling,closed_rhs];cost=np.r_[np.zeros(n),-1.];last=[0.,.5]
 closed=sparse.hstack([HN,sparse.csr_matrix((H.shape[0],1))]);parts=[top,closure,closed,-closed]
 if ceiling is not None:
  ones=sparse.csr_matrix(np.ones((H.shape[0],1)));parts += [sparse.hstack([HN,ones]),sparse.hstack([-HN,ones])];rhs=np.r_[rhs,[float(u/w) for u,w in zip(s['upper'],s['widths'])],[-float(l/w) for l,w in zip(s['lower'],s['widths'])]]
 E=(d['E']@d['P']).tocsr();E.eliminate_zeros();assert E.nnz==0
 return dict(matrix=sparse.vstack(parts,format='csr'),rhs=rhs,cost=cost,bounds=np.c_[np.r_[np.full(n,-np.inf),last[0]],np.r_[np.full(n,np.inf),last[1]]],equality=sparse.csr_matrix((int(d['eq']['optimizer_keep'].sum()),n+1)),equality_rhs=np.zeros(int(d['eq']['optimizer_keep'].sum())),ceiling=ceiling)

def exact_product(M,t):
 tf=list(map(fq,t));out=[]
 for i in range(M.shape[0]):
  b,e=M.indptr[i:i+2];out.append(sum((fq(v)*tf[j] for v,j in zip(M.data[b:e],M.indices[b:e])),F(0)))
 return out

def certificate(s,t):
 v=exact_product(s['H'],t);eta=F(1,2);bad=[];owner=None
 for i,(x,l,u,w) in enumerate(zip(v,s['lower'],s['upper'],s['widths'])):
  q=min(x-l,u-x)/w
  if q<eta:eta=q;owner=s['owners'][i]
  if q<=0:bad.append(i)
 return dict(passed=eta>0,eta=pair(eta),nonpositive=bad,limiting_owner=owner)

def decode(d,origin,drops,t,s):
 cert=certificate(s,t)
 if not cert['passed']:raise ValueError('Exact lifted interval certificate rejected')
 delta=exact_product(d['P'],t);q=[fq(c)+v for c,v in zip(origin.ravel(),delta)];out=np.array(list(map(float,q))).reshape(origin.shape)
 dd=exact_product(d['N'],t);raw=[c+v for c,v in zip(drops,dd)]
 if any(dd):newdrops,raised=joint_round(raw,d['dom'])
 else:newdrops=drops.copy();raised=[]
 ng=gaps(newdrops,d['dom'])[0];out[d['normal_states'],71]=ng;out=canonical(d,out)
 active=d['active'];loaderror=F(0)
 for b in active:
  assert sum(q[b*75+16:b*75+24],F(0))==sum(map(fq,origin[b,16:24]),F(0))
  err=abs(sum(map(fq,out[b,16:24]),F(0))-sum(map(fq,origin[b,16:24]),F(0)));assert err<=8*F(1,2**54);loaderror=max(loaderror,err)
 z=out.ravel()[d['control_feature_rows']];assert np.all(z>=d['lo']) and np.all(z<=d['hi'])
 for i,row in enumerate(d['varying']):
  p=len(d['lo'])+i;change=fq(out.ravel()[row])-fq(origin.ravel()[row]);assert s['closed_lower'][p]<=change<=s['closed_upper'][p]
 assert np.array_equal(out[d['mesh']['fixed']],origin[d['mesh']['fixed']]);assert np.all(out[:,16:24][d['mesh']['inactive']]==0)
 return out,newdrops,dict(certificate=cert,shared_alpha=[1,1],load_sum_rounding=pair(loaderror),raised=raised,drop_state=[pair(x) for x in newdrops],actual_mechanics_evaluated=False)
