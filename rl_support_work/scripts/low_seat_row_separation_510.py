"""Whole-row separation for the unchanged lifted minimax domain.
Preparation utilities only: no optimizer, mechanics, decoder or run wrapper.
"""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
TOP=331495
TOL=1e-7
SPEC=dict(
 scope='NEW bounded monotone whole-row separation method. This file prepares selection and certification arithmetic only; no optimizer or actual candidate.',
 rationale='506 supplied no witness from a simultaneous652529-row LP. Change which original path rows enter each finite master, with an exhaustive separation oracle and monotone cut ledger. This changes the numerical witness method, not another coordinate basis, solver-method retry, or cause/speedup claim.',
 representation='Keep the already-audited504 mixed basis unchanged. Native35auxiliary increments, all68250 two-sided unit-scaled mechanics consistency constraints, all92267 closed finite intervals, every witness interior interval row, all original equalities and auxiliary bounds are ALWAYS in each master. Only the first331495 original path rows may initially be deferred to exhaustive separation.',
 initial_set='At every primary or witness construction add each top row whose ORIGINAL RHS<=0, the exact binary64 sign at ORIGINAL direction zero and epigraph/eta zero. Union with every earlier top cut in the same run. The anchor is an arithmetic selection rule, NOT a feasible path, a finite map, or zero in the affine504 coordinates. No hand-chosen proximity threshold, row scaling change or coefficient deletion.',
 cuts='After an optimal master witness, reconstruct ORIGINAL t=S*z+a unmodified. Evaluate EVERY row of BOTH the full original and full transformed LP. Add ALL omitted top rows with a strictly positive signed residual in EITHER representation, including arbitrarily small positive binary64 residuals. Record both complete residual vectors and sorted global row IDs. Never remove cuts or substitute an internal solver state.',
 eligibility='A master status0 is necessary but insufficient. First require finite unchanged vectors and original/transformed active inequalities, all equalities and bounds<=unchanged1e-7. Failure terminates rather than editing a vector. A full witness requires no omitted positive residual in either representation AND both complete LP violations<=1e-7. Exact original interval eta>0 remains a SEPARATE requirement before each finite map; no subset direction is decoded.',
 proof='In exact arithmetic, every master is a relaxation of the fixed full LP. A master optimum satisfying every deferred full row is also a full optimum: its objective is both a lower bound on and attained by the full problem. Otherwise every unsuccessful separation adds at least one previously absent row, so there are finitely many enlargements. This exact-arithmetic statement does NOT certify the rounded transformed LP as identical to the original; original all-row certification remains mandatory and numerical optimality is not promoted to a physical result.',
 bounded_policy='PROPOSED ONE new run:24major iterations,48actual LP calls TOTAL including every primary/interior separation round,120maps max5 shared dyadics per complete interior witness. Extra cuts consume the SAME48-call budget, not additional calls outside it. Two full witnesses per successful major imply at most24 majors. Reaching the LP cap with outstanding cuts yields NO certified witness, not infeasibility. Every call45s nativehighs-ipm1e-9,40GiBchild/120swall; terminal first nonoptimal/absent/uncertified active witness/resourcefailure, NO retry or timeout-driven trust reduction.',
 nonlinear='After full primary certification compute theta from reconstructed ORIGINAL direction, never the master objective alone; ceiling=(mu+theta)/2. After full interior certification reconstruct ORIGINAL before any dyadic scale, then exact interval certificate and immutable490 finite encoding. SAME effective encoded-feature prediction and actual phi ratio>=.1; trust1..1/32, halve only after actual nonlinear rejection, double<=1 only ratio>.75. No15-root. Changed states require fresh mechanics/J and valid504-style basis proof. No new state is evaluated here.',
 all_criteria='All331888 final and331495 optimizer path rows,156647HARD/noepigraph and174848rate/window labels,1411 original normal/hover primal tests,1366 homogeneous equalities,11906fullcontrolboxes,80155featuretrustintervals,206dropbounds includingpivot,27708old/support/window+1572bounds remain. All8eps/4ULPper20ms/32ULPtarget/4302^-23 reserves and original440 +/-.02box unchanged. Same392constantcontact/timeidentity optimizer-only omission, allfinalchecked.',
 final='Auxiliary/LP/working-set success never admits actual mechanics or execution. All actual signed rows, unit-specific closures,normal1e-13,hover1e-8,kinematic/geometry/clock/window/new-state-work checks plus laterfilter/runtime/physics gates remain. No old rejected/final/internal solver vector initializes this method. Seed would be EXACTsealed472INITIAL actualX/Y/J and490canonicalauxiliary.',
 audit_scope='Saved504/490 full matrices are READ, not rebuilt; new subset selection and separation arithmetic are checked. No prior basis/inverse/incidence/map/root/model/finitework fixture repeated. Witness fixture ceiling is inherited illustrative.09895564659603738, not a solved value.',
 executable_whole_path_wrapper=False,lp_calls=0,finite_maps=0,model_calls=0,root_calls=0,physical_admission=False)

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def seed_rows(rhs,previous=(),top=TOP):
 rhs=np.asarray(rhs);previous=np.asarray(previous,dtype=np.int64)
 if not np.all(np.isfinite(rhs)):raise ValueError('Nonfinite RHS')
 if len(previous) and (previous.min()<0 or previous.max()>=top):raise ValueError('Invalid previous top-row ID')
 topids=np.union1d(np.flatnonzero(rhs[:top]<=0),previous)
 return np.r_[topids,np.arange(top,len(rhs))].astype(np.int64)
def validate_rows(ids,total,top=TOP):
 ids=np.asarray(ids,dtype=np.int64)
 if not np.array_equal(ids,np.unique(ids)) or (len(ids) and (ids[0]<0 or ids[-1]>=total)):raise ValueError('Rows not sorted unique valid IDs')
 if not np.array_equal(ids[ids>=top],np.arange(top,total)):raise ValueError('A mandatory consistency/interval row is absent')
 return ids

def decide(ids,original_residual,transformed_residual,other_violation,status,finite=True,top=TOP):
 ro=np.asarray(original_residual);rt=np.asarray(transformed_residual)
 if ro.shape!=rt.shape or ro.ndim!=1:raise ValueError('Residual shapes disagree')
 ids=validate_rows(ids,len(ro),top);mask=np.zeros(len(ro),dtype=bool);mask[ids]=True
 finite=bool(finite and np.all(np.isfinite(ro)) and np.all(np.isfinite(rt)) and np.isfinite(other_violation))
 active=max(0.,float(other_violation),float(np.max(ro[ids],initial=0)),float(np.max(rt[ids],initial=0))) if finite else None
 full=max(0.,float(other_violation),float(np.max(ro,initial=0)),float(np.max(rt,initial=0))) if finite else None
 new=np.flatnonzero((~mask)&((ro>0)|(rt>0))) if finite else np.array([],dtype=np.int64)
 ok=bool(status==0 and finite and active<=TOL)
 action='terminal' if not ok else ('add_all_violated_rows' if len(new) else ('full_witness' if full<=TOL else 'terminal'))
 nxt=np.union1d(ids,new) if action=='add_all_violated_rows' else ids.copy()
 return dict(action=action,status=int(status),finite=finite,active_violation=active,full_violation=full,omitted_positive_original=int(np.count_nonzero((~mask)&(ro>0))),omitted_positive_transformed=int(np.count_nonzero((~mask)&(rt>0))),new_rows=new.tolist() if action=='add_all_violated_rows' else [],next_rows=nxt.tolist(),full_witness=action=='full_witness')

def evaluate(full,transformed,original_vector,transformed_vector,ids,status):
 w=np.asarray(original_vector);q=np.asarray(transformed_vector)
 ro=full['matrix']@w-full['rhs'];rt=transformed['matrix']@q-transformed['rhs']
 other=0.
 for p,v in [(full,w),(transformed,q)]:
  other=max(other,float(np.max(abs(p['equality']@v-p['equality_rhs']),initial=0)),float(np.max(p['bounds'][:,0]-v,initial=0)),float(np.max(v-p['bounds'][:,1],initial=0)))
 result=decide(ids,ro,rt,other,status,np.all(np.isfinite(w)) and np.all(np.isfinite(q)))
 return result,ro,rt

def fixture_cases():
 # New row-separation decision fixtures only; no old tangent/map fixture.
 z=np.zeros(7);ids=np.array([0,4,5,6]);tiny=np.nextafter(0.,1.)
 cases=[]
 def add(name,ro,rt,other=0.,status=0,finite=True,rows=ids):
  result=decide(rows,ro,rt,other,status,finite,top=4);cases.append(dict(name=name,rows=rows.tolist(),original=np.asarray(ro).tolist(),transformed=np.asarray(rt).tolist(),other=other,status=status,finite=finite,result=result))
 ro=z.copy();ro[2]=.25;add('deferred_original_row',ro,z)
 rt=z.copy();rt[1]=tiny;add('transformed_only_smallest_positive',z,rt)
 ro=z.copy();ro[0]=2e-7;add('active_failure_terminal',ro,z)
 add('nonoptimal_never_certifies',z,z,status=1)
 ro=z.copy();ro[3]=tiny;add('original_smallest_positive_not_waived',ro,z)
 add('complete_zero_residual',z,z)
 ro=z.copy();ro[0]=5e-8;add('unchanged_active_numerical_tolerance',ro,z)
 add('bound_failure_terminal',z,z,other=2e-7)
 union=np.union1d(ids,[1,2]);ro=z.copy();ro[3]=.125;add('monotone_second_cut',ro,z,rows=union)
 add('nonfinite_flag_terminal',z,z,finite=False)
 return cases
