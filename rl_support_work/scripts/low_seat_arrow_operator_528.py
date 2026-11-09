"""Lossless state-local/separator factorization of a weighted normal OPERATOR.

This is not an LP method, an LP solve, a trajectory direction, or an inverse of
the mechanics Jacobian. All inputs are the already saved 522 numerical systems.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy import sparse,linalg
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
SPEC=dict(
 scope='NEW lossless row-local arrow factorization of weighted regularized normal operators for the two SAVED522 projected numeric matrices. Preparation only; no LP algorithm, budget, witness or candidate.',
 inputs='SAVED522 primary/witness and523 proof plus490 coordinate labels. No projection, pair proof, quotient grouping, old local inverse, root, model, finite map or LP repeated. The witness remains an illustrative assembly, not an actual primary-dependent ceiling.',
 partition='Candidate local variables are native rootxy/f/legs15 and COM3 per nonfixed state. Joint g/p/load/drop and all command auxiliaries plus epigraph/eta are separators. If any row touches candidate locals in multiple states, promote ALL candidate-local columns on that row to separators. This deterministic one-pass promotion cannot introduce a new multi-state local row. No variable or row is deleted.',
 exact_structure='Every original stored nonzero is copied into exactly one of B=A[:,local] or C=A[:,separator]. Every row of B has support in at most one state block. Thus B.T*diag(d)*B is exactly block diagonal in real arithmetic for ANY diagonal d. Signs, tiny nonzero coefficients, all rows, bounds, objective and equalities remain unchanged.',
 operator='For strictly positive finite diagonal d,w define K=diag(w)+A.T*diag(d)*A. With local first, G=diag(w_local)+B.T*diag(d)*B is block diagonal, V=B.T*diag(d)*C, T=diag(w_separator)+C.T*diag(d)*C. K is SPD since v.T*K*v=sum(w*v*v)+sum(d*(A*v)^2)>0 for v!=0. Each G block is SPD.',
 factorization='K=[[I,0],[V.T*G^-1,I]]*diag(G,S)*[[I,G^-1*V],[0,I]], S=T-V.T*G^-1*V. This identity is an exact-real algebraic theorem; a rounded Cholesky solve is not an exact inverse. S is SPD by the minimization identity y.T*S*y=min_x (x,y).T*K*(x,y).',
 sparse_action='S*y=w_separator*y+C.T*(d*(C*y-B*solve(G,B.T*(d*(C*y))))). Never form K, V, T, S or a dense separator Schur matrix. Store only B,C and small local Cholesky blocks. RHS condenses to b_separator-C.T*(d*(B*solve(G,b_local))); recovery is solve(G,b_local-B.T*(d*(C*y))).',
 numerical_example='ONE NEW operator-action/RHS example per saved system. Deterministic algorithm-unused separator y and b. d_i=2^(-2*frexp(max(1,sum(abs(A_i)))) exponent), w=1. This dyadic weighting is ONLY an algebra exercise, not LP row scaling, guards, tolerances, an IPM iterate, barrier weights or a physical model. Cholesky factors are of NEW weighted Gram blocks, not old L/K/Q mechanics matrices. No LP or Schur equation is solved.',
 validation='Independent529 must verify every copied coefficient/support, partition and equality/cost/bound provenance exactly; check the new saved factors and actions against the original matrix without repeating the local factorizations/solves. Relative algebra residual threshold1e-11 is not a physical or LP threshold, a directed-rounding proof, or a claim about future solver weights.',
 limitations='This does not specify initialization, feasibility restoration, barrier/centering updates, regularization convergence, a Schur linear solver/preconditioner, stopping/budget or final LP witness construction. A future complete method must account for those and justify its resources before launch. Arbitrary changing weights need fresh factors and residual checks. No speedup, conditioning improvement, timeout cause or feasibility claim.',
 gates='522 exact-real projection differs from its rounded numeric matrix. Future retained LP coordinates stay unchanged; exact endpoint/midpoint binary64 backsubstitution, all original490 checks1e-7, actual primary-derived ceiling, exact interval eta>0 and immutable490 encoding remain mandatory. Every actual mechanics/geometry/filter/runtime/physical/PPO gate remains. No command-auxiliary Fourier-Motzkin elimination is attempted.',
 lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,new_actual_path=False,physical_admission=False)

def partition(A,retained,extra_rows):
 labels=np.full(A.shape[1],-1,dtype=np.int64)
 for j,c in enumerate(retained):
  if 10540<=c<78790:
   f=int(extra_rows[c-10540]);k=f%75
   if k in (0,1,2,*range(4,16),72,73,74):labels[j]=f//75
 initial=labels.copy();promoted=np.zeros(A.shape[1],dtype=bool);cross=[]
 for i in range(A.shape[0]):
  p,q=A.indptr[i:i+2];cols=A.indices[p:q][A.data[p:q]!=0];loc=cols[labels[cols]>=0]
  if len(loc) and np.any(labels[loc]!=labels[loc[0]]):promoted[loc]=True;cross.append(i)
 labels[promoted]=-1
 local=np.flatnonzero(labels>=0);local=local[np.lexsort((local,labels[local]))];sep=np.flatnonzero(labels<0)
 states,starts=np.unique(labels[local],return_index=True);ptr=np.r_[starts,len(local)]
 B=A[:,local].tocsr();C=A[:,sep].tocsr();owner=np.full(A.shape[0],-1,dtype=np.int64)
 block_of=np.repeat(np.arange(len(states)),np.diff(ptr))
 for i in range(B.shape[0]):
  p,q=B.indptr[i:i+2];v=B.indices[p:q][B.data[p:q]!=0]
  if len(v):
   group=block_of[v];assert np.all(group==group[0]);owner[i]=group[0]
 return B,C,dict(initial_labels=initial,labels=labels,promoted=np.flatnonzero(promoted),cross_rows=np.array(cross,dtype=np.int64),local=local,separator=sep,states=states,block_ptr=ptr,row_owner=owner)

def operator_example(A,B,C,part,out):
 row_l1=np.asarray(abs(A).sum(axis=1)).ravel();exponent=np.frexp(np.maximum(1.,row_l1))[1]
 weights=np.ldexp(np.ones(len(exponent)),-2*exponent);assert np.all(np.isfinite(weights)) and np.all(weights>0)
 ptr=part['block_ptr'];owner=part['row_owner'];factors=[];factor_data=[];factor_ptr=[0]
 # Sparse block row selection avoids scanning the full matrix once per state.
 order=np.argsort(owner,kind='stable');valid=order[owner[order]>=0];counts=np.bincount(owner[valid],minlength=len(ptr)-1);rp=np.r_[0,np.cumsum(counts)]
 grams=[]
 for k,(lo,hi) in enumerate(zip(ptr[:-1],ptr[1:])):
  rows=valid[rp[k]:rp[k+1]];b=B[rows,lo:hi];G=np.eye(hi-lo)+(b.T@b.multiply(weights[rows,None])).toarray()
  F=linalg.cholesky(G,lower=True,check_finite=True);factors.append(F);grams.append(G)
  factor_data.extend(F.ravel());factor_ptr.append(len(factor_data))
 def solve(v):
  z=np.empty_like(v)
  for k,(lo,hi) in enumerate(zip(ptr[:-1],ptr[1:])):z[lo:hi]=linalg.cho_solve((factors[k],True),v[lo:hi],check_finite=True)
  return z
 n=C.shape[1];y=((np.arange(n)%17)-8).astype(float)/16
 bl=((np.arange(B.shape[1])%13)-6).astype(float)/16;bg=((np.arange(n)%19)-9).astype(float)/16
 cy=C@y;u=solve(B.T@(weights*cy));action=y+C.T@(weights*(cy-B@u))
 q=solve(bl);rhs=bg-C.T@(weights*(B@q));recovered=q-u
 assert all(np.all(np.isfinite(v)) for v in [action,rhs,recovered])
 np.savez_compressed(out/'operator_example.npz',weights=weights,weight_exponents=exponent,separator_vector=y,local_rhs=bl,separator_rhs=bg,local_response=u,local_rhs_response=q,schur_action=action,condensed_rhs=rhs,recovered_local=recovered,factor_data=np.asarray(factor_data),factor_ptr=np.asarray(factor_ptr))
 return dict(local_blocks=len(factors),maximum_local_size=int(max(np.diff(ptr))),local_factor_entries=len(factor_data),local_cholesky_factorizations=len(factors),local_triangular_rhs_solves=2*len(factors),new_operator_examples=1,full_normal_or_schur_matrix_assembled=False,schur_equation_solved=False,weights_scope='fixed algebra-only example; not barrier weights or LP scaling')
