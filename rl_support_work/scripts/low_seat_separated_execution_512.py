"""Bounded whole-row separation, with exhaustive original and transformed certification."""
from low_seat_mixed_execution_506 import *
from low_seat_row_separation_510 import seed_rows,validate_rows,evaluate as separate_evaluate,SPEC as SEPARATION_SPEC,TOP
SPEC=dict(SPEC)
SPEC.update(scope=__doc__,row_separation=SEPARATION_SPEC,solver_failure_policy='Terminal first nonoptimal/absent/active-uncertified master or resource failure; no retry. Successful but incomplete master adds ALL omitted positive rows. All cut rounds consume48 total LP calls.',max_actual_lps=48,rationale='A monotone whole-row separation algorithm with all mechanics-consistency signs and finite intervals always in the master, full original/transformed row certification, and explicit cut ledger. Not a new basis or solver retry.',executable_whole_path_wrapper=True)

class Separator:
 def __init__(self):
  self.top=np.array([],dtype=np.int64);self.calls=0;self.assemblies=0;self.stages=0;self.cut_rounds=0;self.added_rows=0;self.reconstructions=0;self.full_witnesses=0
 def solve(self,folder,d,b,s,S,a,ceiling=None):
  folder.mkdir(exist_ok=False);p=assemble(d,b,s,ceiling);self.stages+=1
  for name in ['matrix','equality']:sparse.save_npz(folder/(name+'.npz'),p[name])
  np.savez_compressed(folder/'arrays.npz',**{k:p[k] for k in ['rhs','cost','bounds','equality_rhs']});write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
  T=folder/'transformed';T.mkdir();V=sparse.block_diag([S,sparse.csr_matrix([[1.]])],format='csr');z=dict(p);z['matrix']=(p['matrix']@V).tocsr();z['rhs']=p['rhs']-p['matrix'][:,:-1]@a
  for name in ['matrix','equality']:sparse.save_npz(T/(name+'.npz'),z[name])
  np.savez_compressed(T/'arrays.npz',**{k:z[k] for k in ['rhs','cost','bounds','equality_rhs']})
  np.save(folder/'previous_top_rows.npy',self.top)
  ids=seed_rows(p['rhs'],self.top);self.top=ids[ids<TOP].copy();rounds=[];w=None;start_calls=self.calls;start_assemblies=self.assemblies
  reason='48-call total budget exhausted before a complete witness';last=dict(status=-48,message=reason,candidate=False,actual_lp_call=False,iterations=0,crossover_iterations=0,worker_actual_exit=None,worker_wall_timeout=False,elapsed_s=0.,worker_elapsed_s=0.)
  while self.calls<48:
   R=folder/f'round_{len(rounds):02d}';R.mkdir();validate_rows(ids,len(p['rhs']));np.save(R/'row_ids.npy',ids)
   sparse.save_npz(R/'matrix.npz',z['matrix'][ids]);sparse.save_npz(R/'equality.npz',z['equality']);np.savez_compressed(R/'arrays.npz',rhs=z['rhs'][ids],cost=z['cost'],bounds=z['bounds'],equality_rhs=z['equality_rhs']);self.assemblies+=1
   start=time.monotonic();timed=False
   with (R/'worker.log').open('w') as log:
    proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_separated_lp_512.py')),str(R)],stdout=log,stderr=subprocess.STDOUT)
    try:rc=proc.wait(timeout=120)
    except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
   called=(R/'scipy_started.json').exists();self.calls+=int(called)
   meta=read(R/'result.json') if (R/'result.json').exists() else dict(status=-120 if timed else -30,message='Child wallcap' if timed else 'Child exited before result',candidate=False,success=False,iterations=0,crossover_iterations=0)
   meta.update(actual_lp_call=called,worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-start,method='highs-ipm',selected_rows=len(ids),selected_path_rows=int((ids<TOP).sum()),master_nnz=int(z['matrix'][ids].nnz),full_rows=len(p['rhs']),total_actual_lps=self.calls)
   meta.setdefault('elapsed_s',meta['worker_elapsed_s']);last=meta
   if rc!=0 or timed or meta['status']!=0 or not meta['candidate']:
    decision=dict(action='terminal',reason='Absent/nonoptimal master or resource failure; no internal state selection',full_witness=False);reason=decision['reason']
   else:
    q=np.load(R/'witness.npy');ow=np.r_[S@q[:-1]+a,q[-1]];np.save(R/'original_witness.npy',ow);self.reconstructions+=1
    decision,ro,rt=separate_evaluate(p,z,ow,q,ids,meta['status']);np.savez_compressed(R/'full_residuals.npz',original=ro,transformed=rt)
    if decision['action']=='full_witness':
     w=ow;np.save(folder/'witness.npy',w);np.save(T/'witness.npy',q);self.full_witnesses+=1;reason='Exhaustive original and transformed full-row certification passed'
    elif decision['action']=='add_all_violated_rows':
     ids=np.asarray(decision['next_rows'],dtype=np.int64);self.top=ids[ids<TOP].copy();self.cut_rounds+=1;self.added_rows+=len(decision['new_rows']);reason='48-call total budget exhausted with deferred violations' if self.calls>=48 else 'Monotone cut enlargement'
    else:reason='Active/equality/bound certification failed; returned vector unchanged'
   write(R/'decision.json',decision);write(R/'worker_exit.json',meta);rounds.append(dict(folder=str(R),worker=meta,decision=decision['action'],new_rows=len(decision.get('new_rows',[]))))
   if decision['action']!='add_all_violated_rows':break
  result=dict(last);result.update(candidate=w is not None,transformed_candidate=w is not None,terminal_failure=w is None,reason=reason,actual_lp_calls=self.calls-start_calls,master_assemblies=self.assemblies-start_assemblies,rounds=rounds,linear_pass=w is not None,full_original_and_transformed_certified=w is not None,transformed_nnz=z['matrix'].nnz)
  if w is not None:result.update(linear_violation=violation(p,w),transformed_linear_violation=violation(z,np.load(T/'witness.npy')),objective=float(p['cost']@w),reconstruction_unmodified=True)
  np.save(folder/'final_top_rows.npy',self.top);write(folder/'solver.json',result);return w,result
