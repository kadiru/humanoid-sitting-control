"""New complete-clock row/Schur-map preparation, not a trajectory optimizer.
Reuses saved444 local derivatives. Twelve NEW perturbed14-equation roots test
the joint f/p/persistent-load retraction; no440 assessment or old finite work is
repeated. Original27708 constraints and all prior inward buffers are retained.
"""
import json,datetime,time
import numpy as np
from scipy import sparse
from low_seat_clock_rows_446 import make_rows,D,read,BUFFER
from low_seat_clock_transcription_444 import Mechanics,load_inputs,sha,FLOOR
O=D/'clock_row_preparation_446'

def main():
 launch=read(D/'clock_row_preparation_launch_446.json')
 assert launch['audit445_actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 O.mkdir(exist_ok=False);start=time.monotonic();a=make_rows();r=a['rows'];features=a['features'];g=a['graph'];K=len(a['canonical']);M=r.matrix();b=np.array(r.b);scale=np.array(r.scale);soft=np.array(r.soft);groups=np.array(r.group);rows=M@features.ravel()+b
 sparse.save_npz(O/'all_rows_matrix.npz',M);sparse.save_npz(O/'original_rows_matrix.npz',a['old_matrix'])
 np.savez_compressed(O/'row_arrays.npz',constant=b,scale=scale,keep=a['keep'],soft=soft,groups=groups,rows=rows,features=features,old_constant=a['old_constant'],old_tightening=a['old_tightening'],old_raw=a['old_raw'],old_prior=a['old_prior'],node_states=a['node_states'])
 src=dict(np.load(D/'contact_lift_440/states.npz'))
 raw_error=float(abs(a['old_raw']-src['node_raw']).max());prior_error=float(abs(a['old_prior']-src['node_prior']).max());old_keep=a['keep'][:27708]
 buffered=np.r_[rows[:27708][old_keep]/scale[:27708][old_keep],rows[27708:29280]]
 buffer_error=float(abs(buffered-src['node_buffered']).max())
 assert max(raw_error,prior_error,buffer_error)<2e-10,(raw_error,prior_error,buffer_error)
 # Implicit14-coordinate root tangent. f,p and persistent loads remain joint
 #coordinates; no exact gap profile controls f at intermediate states.
 dep=np.array([0,1]+list(range(4,16)));control=np.array([2,3]+list(range(16,24)));J=g['canonical_J'];A=J[:,:14][:,:,dep];B=J[:,:14][:,:,control]
 tangent=-np.linalg.solve(A,B);T=np.zeros((K,24,10));T[:,control,:]=np.eye(10);T[:,dep,:]=tangent
 G=np.concatenate([T,np.einsum('noi,nic->noc',J,T)],axis=1)
 eq_error=float(abs(G[:,24:38]).max());assert eq_error<2e-11,eq_error
 lower=np.tile(np.r_[0.,0.,np.full(8,FLOOR)],(K,1));upper=np.ones((K,10));reference=g['reference'][:,control]
 for i in range(K):
  if g['fixed'][i]:lower[i]=reference[i];upper[i]=reference[i]
  else:
   lower[i,2:][g['inactive'][i]]=0;upper[i,2:][g['inactive'][i]]=0
 free=np.flatnonzero((lower!=upper).ravel());assert np.all(lower<=upper)
 #No fake positive seed feasibility:440's phase/rate failures remain as saved.
 fullG=sparse.block_diag(list(G),format='csr');globalJ=(M@fullG)[:,free].tocsr()
 sparse.save_npz(O/'global_jacobian.npz',globalJ)
 #The14 foot/moment equations are solved by the proposed retraction. Total
 #normal and hover remain explicit equalities of the reduced transcription.
 eqri=list(range(K))+[K];eqci=[i*75+38 for i in range(K)]+[int(g['label_to_state'][1450])*75+71]
 E=sparse.coo_matrix((np.r_[np.ones(K),1/.012],(eqri,eqci)),shape=(K+1,K*75)).tocsr();eb=np.r_[np.zeros(K),-1.]
 equality=E@features.ravel()+eb;EJ=(E@fullG)[:,free].tocsr();eqkeep=np.r_[~g['fixed'],True]
 sparse.save_npz(O/'equality_matrix.npz',E);sparse.save_npz(O/'equality_jacobian.npz',EJ)
 np.savez_compressed(O/'equality_arrays.npz',constant=eb,values=equality,optimizer_keep=eqkeep)
 np.savez_compressed(O/'coordinate_map.npz',G=G,T=T,lower=lower,upper=upper,free=free,reference=reference,dependent=dep,control=control,condition=np.linalg.cond(A))
 # Five mechanism locations plus hover; endpoint fixed poses are not perturbed.
 s=load_inputs();indices=np.array([1366,93,912,604,1300,1450]);oracle=Mechanics(s,indices);x=s['X'][indices];jj=np.load(D/'clock_transcription_preparation_444/local_blocks.npz')['J'][indices]
 dirs=np.zeros((6,10));dirs[:,:2]=np.c_[.03*np.sin(indices),.01*np.cos(indices)];dirs[:,2:]=.01*np.sin(indices[:,None]+np.arange(8)[None,:])
 for i,label in enumerate(indices):
  active=s['z']['weights'][label]>0;dirs[i,2:][~active]=0;dirs[i,2:][active]-=dirs[i,2:][active].mean()
 smallT=np.zeros((6,24,10));smallT[:,control,:]=np.eye(10);smallT[:,dep,:]=-np.linalg.solve(jj[:,:14][:,:,dep],jj[:,:14][:,:,control]);expected=np.einsum('noi,nic,nc->no',jj,smallT,dirs)
 completed=[];h=1e-5
 for sign in [1.,-1.]:
  trial=x.copy();trial[:,control]+=sign*h*dirs;iterations=0
  for it in range(10):
   val=oracle.values(trial);res=val[:,:14]
   if np.max(abs(res))<=1e-13:break
   jac=oracle.jacobian(trial);delta=np.linalg.solve(jac[:,:14][:,:,dep],-res[...,None])[...,0];norm=np.linalg.norm(res,axis=1);accepted=np.max(abs(res),axis=1)<=1e-13
   for half in range(9):
    candidate=trial.copy();active=np.flatnonzero(~accepted);candidate[active[:,None],dep]+=delta[active]*2.**(-half);rv=oracle.values(candidate)[:,:14];take=(~accepted)&(np.linalg.norm(rv,axis=1)<norm);trial[take]=candidate[take];accepted|=take
    if accepted.all():break
   assert accepted.all(),'new fixture root descent cap';iterations=it+1
  else:raise RuntimeError('new fixture root iteration cap')
  value=oracle.values(trial);entry=dict(sign=sign,iterations=iterations,maximum_equation_residual=float(abs(value[:,:14]).max()),X=trial.tolist(),Y=value.tolist())
  with (O/'new_root_fixtures.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
  completed.append(entry)
 fd=(np.array(completed[0]['Y'])-np.array(completed[1]['Y']))/(2*h);direction_error=float(abs(fd-expected).max())
 np.savez_compressed(O/'new_root_directional.npz',indices=indices,directions=dirs,expected=expected,finite=fd,h=h)
 assert direction_error<2e-6,direction_error
 vector=np.sin(np.arange(len(free))*.13);delta=(fullG[:,free]@vector).reshape(K,75)
 graph_error=float(abs(globalJ@vector-M@delta.ravel()).max());assert graph_error<2e-10,graph_error
 (O/'groups.json').write_text(json.dumps({name:dict(rows=int((groups==name).sum()),negative=int((rows[groups==name]<0).sum()),minimum=float(rows[groups==name].min())) for name in sorted(set(groups.tolist()))},indent=2))
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,passed=True,total_inequality_rows=M.shape[0],retained_optimizer_rows=int(a['keep'].sum()),explicit_equality_rows=K+1,retained_equality_rows=int(eqkeep.sum()),old_original_rows=27708,old_coordinate_bound_rows=1572,all_original_rows_replay_error=raw_error,all_prior_rows_replay_error=prior_error,old_buffered_rows_replay_error=buffer_error,state_blocks=K,free_joint_phase_load_coordinates=len(free),root_dependent_coordinates_per_nonfixed_state=14,maximum_root_condition=float(np.linalg.cond(A).max()),maximum_implicit_equation_tangent=eq_error,new_root_directional_error=direction_error,unchanged_directional_bound=2e-6,new_perturbed_root_states=12,full_path_roots_repeated=0,sparse_chain_error=graph_error,minimum_scaled_row=float(np.min(rows[a['keep']]/scale[a['keep']])),minimum_soft_row=float(np.min(rows[soft]/scale[soft])),minimum_hard_row=float(np.min(rows[a['keep']&~soft]/scale[a['keep']&~soft])),new_candidate=False,solver_specification_frozen=False,optimizer_launch_allowed=False,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,physical_admission=False,elapsed_s=time.monotonic()-start)
 (O/'preparation.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
