"""Finite JOINT decoder and bounded 15-equation lift for NEW free-gap path472.

The audited470 assembly is used at each actual new state. Exact normal drops468,
conserved bounded loads460 and scalar orientation charts are recentered for
every trial. Prehover gap is a free nominal coordinate; dependent f retains all
old boxes and hard phase rows. No old trial is selected or optimizer continued.
"""
import json,time
import numpy as np
from scipy.optimize import linprog
from low_seat_free_gap_solver_470 import *
from low_seat_gap_coordinates_v2_467 import EQS

def start_targets(d):
 saved=read(D/'exact_gap_chart_preparation_468/layout.json')
 drops=list(map(unpack,saved['initial']))
 target=d['initial_target'].copy()
 target[np.asarray(d['gap_layout']['normal_states']),0]=gaps(drops,d['gap_domain'])[0]
 return target,drops

def map_interval(d,a):
 radius=SPEC['coordinate_domain_radius']*a['trust']
 lo=np.maximum(d['lower'],a['origin']-radius);hi=np.minimum(d['upper'],a['origin']+radius)
 if a['mode']=='correction':
  lo=np.maximum(lo,a['z']-.25*radius);hi=np.minimum(hi,a['z']+.25*radius)
 return lo,hi

def decode_joint(d,a,tangent):
 target=a['target'].copy();lo,hi=map_interval(d,a)
 pos={int(v):i for i,v in enumerate(d['free'])};journal=[]
 for group in d['map_groups']:
  b=group['state'];kind=group['kind'];cols=group['columns'];t=tangent[cols]
  if kind=='gap':
   target[b,0]=float(fq(target[b,0])+fq(t[0]))
   journal.append(dict(state=b,kind=kind))
  else:
   if kind=='orientation':slots=[1];pivot=-1
   else:
    pivot_corner=group['entries'][0]['pivot']
    slots=sorted([2+x['index'] for x in group['entries']]+[2+pivot_corner])
    pivot=slots.index(2+pivot_corner)
    assert [s-2 for s in slots if s!=2+pivot_corner]==[x['index'] for x in group['entries']]
   ids=[pos[b*10+s] for s in slots]
   out,alpha,ds,q=decode_group(target[b,slots],t,lo[ids],hi[ids],pivot)
   target[b,slots]=out
   journal.append(dict(state=b,kind=kind,slots=slots,indices=ids,pivot=pivot,alpha=pair(alpha)))
 drops,info=map_drops(a['drops'],tangent[d['normal_columns']],d['gap_domain'])
 target[np.asarray(d['gap_layout']['normal_states']),0]=gaps(drops,d['gap_domain'])[0]
 assert np.array_equal(target[d['g']['fixed']],a['target'][d['g']['fixed']])
 assert np.all(target[:,2:10][d['g']['inactive']]==0)
 return target,drops,dict(groups=journal,drop_alpha=pair(info['alpha']),drop_rho=pair(info['rho']),raised=info['raised'],drops=[pair(v) for v in drops])

class GapRetraction(base.Retraction):
 def __init__(self,d):
  super().__init__(d);self.counts.update(initial_lifts=0,finite_maps=0)
 def lift(self,x,target,folder,initial=False):
  """Only dependent rootxy/f/legs change. Fixed state anchors never move."""
  if initial:self.counts['initial_lifts']+=1
  else:self.counts['trial_paths']+=1
  d=self.d;x=x.copy();x[:,3]=target[:,1];x[:,16:24]=target[:,2:10]
  x[d['g']['fixed']]=d['initialX'][d['g']['fixed']]
  np.savez_compressed(folder/'root_initial.npz',X=x,target=target)
  history=[];reason='root update cap';passed=False
  for it in range(SPEC['root_max_updates']+1):
   y=self.val(x);res=np.c_[y[:,:14],y[:,47]-target[:,0]]
   err=np.max(abs(res),axis=1);bad=(err>SPEC['root_absolute_tolerance'])&~d['g']['fixed']
   entry=dict(iteration=it,maximum=float(err.max()),bad_states=int(bad.sum()),halvings=[]);history.append(entry)
   if not np.any(bad):
    passed=bool(err.max()<=SPEC['root_absolute_tolerance']);reason='converged' if passed else 'fixed-anchor root mismatch';break
   if it==SPEC['root_max_updates']:break
   j=self.jac(x)
   try:step=np.linalg.solve(j[:,EQS][:,:,DEP],-res[:,:,None])[:,:,0]
   except np.linalg.LinAlgError:reason='singular15-equation Jacobian';break
   norm=np.linalg.norm(res,axis=1);accepted=~bad
   for h in range(SPEC['root_max_halvings']):
    ids=np.flatnonzero(~accepted);candidate=x.copy();candidate[ids[:,None],DEP]+=step[ids]*2.**(-h)
    yy=self.val(candidate);rr=np.c_[yy[:,:14],yy[:,47]-target[:,0]]
    take=(~accepted)&(np.linalg.norm(rr,axis=1)<norm)
    entry['halvings'].append(dict(half=h,accepted_states=np.flatnonzero(take).tolist()))
    x[take]=candidate[take];accepted|=take
    if accepted.all():break
   self.counts['root_updates']+=1
   if not accepted.all():
    reason='root line-search cap';y=self.val(x);break
  assert np.array_equal(x[:,3],target[:,1]) and np.array_equal(x[:,16:24],target[:,2:10])
  assert np.array_equal(x[d['g']['fixed']],d['initialX'][d['g']['fixed']])
  (folder/'root_journal.json').write_text(json.dumps(history))
  np.savez_compressed(folder/'root_final.npz',X=x,Y=y,target=target)
  return dict(passed=passed,reason=reason,X=x,Y=y,root_history=history)
 def trial(self,a,tangent,folder):
  self.counts['finite_maps']+=1
  assert self.counts['finite_maps']<=SPEC['max_consecutive_load_roundings']
  target,drops,record=decode_joint(self.d,a,tangent)
  (folder/'mapped.json').write_text(json.dumps(record))
  np.savez_compressed(folder/'mapped.npz',target=target,tangent=tangent)
  x=a['X']+np.einsum('nic,nc->ni',a['T'],target-a['target'])
  result=self.lift(x,target,folder)
  result.update(target=target,drops=drops)
  return result

def solve_new_lp(a,folder):
 save_assembly(a,folder);start=time.monotonic()
 if np.any(a['full_step_lower']>a['full_step_upper']):
  res=None;meta=dict(status=-10,message='empty full-control correction box',candidate=False)
 else:
  res=linprog(a['cost'],A_ub=a['matrix'],b_ub=a['rhs'],A_eq=a['equality'],b_eq=a['equality_rhs'],bounds=a['bounds'],method=SPEC['lp_method'],options=dict(time_limit=SPEC['lp_time_limit_s'],primal_feasibility_tolerance=SPEC['lp_primal_dual_tolerance'],dual_feasibility_tolerance=SPEC['lp_primal_dual_tolerance']))
  meta=dict(status=int(res.status),message=str(res.message),candidate=res.x is not None,success=bool(res.success),iterations=int(res.nit))
  if res.x is not None:
   v=res.x;violation=max(0.,float((a['matrix']@v-a['rhs']).max()),float(abs(a['equality']@v-a['equality_rhs']).max()),float((a['bounds'][:,0]-v).max()),float((v-a['bounds'][:,1]).max()))
   meta.update(independent_linear_violation=violation,linear_pass=violation<=SPEC['independent_linear_bound'],objective=float(a['cost']@v));np.save(folder/'witness.npy',v)
 meta.update(elapsed_s=time.monotonic()-start,mode=a['mode'],merit_ceiling=a['merit_ceiling'])
 (folder/'solver.json').write_text(json.dumps(meta,indent=2));return res,meta

def accepted_state(d,X,Y,r,e,target):
 s=summary(d,X,Y,r,e);gaperr=float(abs(Y[:,47]-target[:,0]).max())
 valid=bool(s['hard_pass'] and s['monotone_phase_pass'] and s['coordinate_bound_minimum']>=0 and s['fixed_states_exact'] and s['inactive_exact_zero'] and s['minimum_corner_weight']>=0 and np.max(abs(e[:-1]))<=1e-13 and abs(e[-1])<=1e-8 and gaperr<=SPEC['root_absolute_tolerance'])
 return s,valid,gaperr
