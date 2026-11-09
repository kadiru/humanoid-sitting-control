"""Independent SAVED full-size solver evidence audit; no linear/LP solves."""
from pathlib import Path
import numpy as np,json
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def audit_solver(T):
 r=read(T/'result.json');stats=r.get('statistics',{});proof=read(T/'normalization_proof.json') if (T/'normalization_proof.json').exists() else None
 if proof is None:
  assert r['status']!=0 and not r['candidate'] and not (T/'witness.npy').exists();return dict(passed=True,input_failure_without_witness=True)
 P=Path(proof['source']);ar=dict(np.load(P/'arrays.npz'));A=sparse.load_npz(P/'matrix.npz').tocsr();A0=sparse.load_npz(P/'cone_original.npz').tocsr();b=ar['rhs'];c=ar['cost'];b0=ar['original_rhs'];c0=ar['original_cost'];local=ar['local'];sep=ar['separator'];bp=ar['block_ptr']
 import hashlib
 for p,h in proof['source_input_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 assert proof['passed']
 if proof['inherited']:
  assert read(D/'schur_hsd_audit_531/independent_audit.json')['passed'] and proof['source_proof_sha256']==hashlib.sha256((D/'schur_hsd_audit_531/SHA256.json').read_bytes()).hexdigest()
 factors=dict(np.load(T/'factors.npz'));ptr=factors['ptr'];B=A[:,local].tocsc();factor_error=0.
 assert len(ptr)-1==stats['local_cholesky_blocks'] and len(ptr)<=len(bp)
 for k,(lo,hi) in enumerate(zip(bp[:-1],bp[1:])):
  if k>=len(ptr)-1:break
  rows=np.unique(B.indices[B.indptr[lo]:B.indptr[hi]]);block=B[rows,lo:hi].toarray();G=np.eye(hi-lo)+block.T@block;L=factors['data'][ptr[k]:ptr[k+1]].reshape(hi-lo,hi-lo)
  err=float(np.max(abs(L@L.T-G))/max(1.,np.max(abs(G))));factor_error=max(factor_error,err);assert err<=1e-11
 linear=read(T/'linear_metrics.json');assert len(linear)==stats['K_solves'] and sum(x['CG_iterations'] for x in linear)==stats['CG_iterations']
 assert all(x['CG_iterations']<=256 for x in linear) and stats['CG_iterations']<=16384 and stats['DR_iterations']<=4096
 Kres=None
 if (T/'last_linear.npz').exists():
  z=dict(np.load(T/'last_linear.npz'));assert int(z['solve_number'])==len(linear)
  if 'solution' in z:
   x=z['solution'];Kres=float(np.linalg.norm(x+A.T@(A@x)-z['rhs']));assert Kres==float(z['residual']) and Kres==linear[-1]['residual']
   assert linear[-1]['completed']==bool(np.isfinite(Kres) and Kres<=float(z['epsilon']))
   if r['message']=='full recovered K residual failed':assert Kres>float(z['epsilon']) or not np.isfinite(Kres)
 cachedres=None
 if (T/'cached_q.npz').exists():
  cache=dict(np.load(T/'cached_q.npz'));zx=cache['zx'];zy=cache['zy'];cachedres=float(np.linalg.norm(np.r_[zx+A.T@zy-c,zy-A@zx-b]))
  if r.get('terminal_stage')=='DR':assert cachedres<=1e-12
  if r['message']=='cached M inverse residual failed':assert cachedres>1e-12 or not np.isfinite(cachedres)
 ledger=read(T/'iteration_metrics.json');assert len(ledger)<=stats['DR_iterations']
 for i,x in enumerate(ledger):assert x['k']==i and x['epsilon']==2.**-20/(i+1)**2 and x['resolvent_residual']<=x['epsilon']
 finalres=None
 if (T/'last_DR.npz').exists():
  z=dict(np.load(T/'last_DR.npz'));u=z['old_u'];v=z['old_v'];t=z['tilde'];n=A.shape[1];m=A.shape[0];rr=u+v
  full=np.r_[t[:n]+A.T@t[n:n+m]+c*t[-1]-rr[:n],t[n:n+m]-A@t[:n]+b*t[-1]-rr[n:n+m],t[-1]-c@t[:n]-b@t[n:n+m]-rr[-1]];finalres=float(np.linalg.norm(full));assert finalres<=float(z['epsilon'])
  un=t-v;un[n:]=np.maximum(un[n:],0.);vn=v-t+un;assert np.array_equal(un,z['u']) and np.array_equal(vn,z['v'])
  if int(z['k'])==0:assert np.all(u[:-1]==0) and np.all(v[:-1]==0) and u[-1]==v[-1]==1
  if r['candidate']:
   q=dict(np.load(T/'candidate.npz'));assert un[-1]>0
   x=np.ldexp(un[:n]/un[-1],ar['column_power']);y=np.ldexp(un[n:n+m]/un[-1],ar['row_power']);s=np.ldexp(vn[n:n+m]/un[-1],-ar['row_power'])
   assert np.array_equal(x,q['x']) and np.array_equal(y,q['y']) and np.array_equal(s,q['slack']) and np.array_equal(x,np.load(T/'witness.npy'))
   assert all(np.all(np.isfinite(a)) for a in [x,y,s]) and np.all(y>=0) and np.all(s>=0)
   assert max(np.max(abs(A0@x+s-b0)),np.max(abs(A0.T@y+c0)),abs(c0@x+b0@y))<=1e-9 and np.max(A0@x-b0)<=1e-7
 else:assert not r['candidate']
 if not r['candidate']:assert not (T/'witness.npy').exists() and not (T/'candidate.npz').exists() and r['status']!=0
 out=dict(passed=True,new_Gram_factors_verified=len(ptr)-1,factor_product_relative_error=factor_error,last_K_residual=Kres,cached_M_residual=cachedres,last_DR_residual=finalres,DR_steps_scalar_checked=len(ledger),full_saved_DR_transitions_checked=int(finalres is not None),all_DR_vectors_replayed=False,solver_calls=0,local_factorizations=0,local_solves=0,physical_admission=False,scope='Saved factors/last K equation/last DR transition and every bounded scalar ledger; no old test, solve or factor repeated. Full intermediate vectors deliberately not retained.')
 return out
