"""ONE complete persistent-contact constrained lift; no trajectory optimizer.
All1467 labeled clock/half-clock/vertex records are constructed and screened.
No actual forward-filter or physics call is permitted by this script.
"""
import json,datetime,time
from pathlib import Path
import numpy as np
from low_seat_contact_lift_438 import *
from low_seat_window_envelope_398 import envelope
O=D/'contact_lift_440'
def old_rows(s,rs,commands,torques,coms):
 ini=s['initial'];ix=np.r_[np.arange(100),np.arange(101,116)]+1351;rr=[rs[i] for i in ix];C=commands[ix][:,s['moving']];tau=torques[ix][1:];Q=np.array([r['q'] for r in rr]);root=np.array([r['root'] for r in rr]);f=np.array([r['phase'] for r in rr]);p=np.array([r['orientation_phase'] for r in rr]);gap=np.array([r['measured_gap'] for r in rr]);com=coms[ix];dt=np.array([float(b-a) for a,b in zip(s['node_times'][:-1],s['node_times'][1:])]);dd=np.diff(C,axis=0)
 W=np.zeros((115,8))
 for k,r in enumerate(rr):
  for j,dest in enumerate(s['mapping'][k]):
   if dest>=0:W[k,j]=r['weights'][dest]
 source=read(D/'clearance_lift_path_330/results.json')['dense']['path'];X=np.zeros((114,23));X[:,:2]=root[1:,:2];X[:,2]=f[1:];X[:,3]=p[1:];X[:,4:16]=Q[1:][:,s['legs']];X[:103,21:23]=(com[1:104,:2]-np.array([v['com'][:2] for v in source[1:104]]))/.1
 X[:,16:21]=np.einsum('nki,nk->ni',ini['null'],W[1:]-ini['base_weights']-np.einsum('nki,ni->nk',ini['wcom'],X[:,21:23]))
 dep=np.array([0,1]+list(range(4,16)));lo=ini['command_lo'];hi=ini['command_hi'];rates=ini['rates'];eff=ini['effort']
 local=np.c_[C[1:]-lo,hi-C[1:],eff-tau,eff+tau,W[1:],1-W[1:]]
 bounds=np.c_[X[:103,dep]-ini['lower'][:103,dep],ini['upper'][:103,dep]-X[:103,dep]]
 phase=np.diff(np.c_[f[:105],p[:105]],axis=0).ravel();upper=(hi-C[1:])*dt[:,None]-.18*dd;lower=(C[1:]-lo)*dt[:,None]+.18*dd
 support=-(ini['support_h'][:,:2]*com[1+ini['support_k'],:2]).sum(1)-ini['support_h'][:,2]-.005
 raw=np.r_[local.ravel(),bounds.ravel(),phase,(rates*dt[:,None]-dd).ravel(),(rates*dt[:,None]+dd).ravel(),upper.ravel(),lower.ravel(),gap[1:99]-.012,.012-gap[100:105],gap[1:105]+1e-8,gap[65:104]-gap[66:105],.006*dt[99:104]-(gap[99:104]-gap[100:105]),0.,support]
 assert len(raw)==20284
 tighten=ini['tighten'][:20284].copy();ulp=ini['ulp']
 for off in [12212,14036]:tighten[off:off+1824]=(dt[:,None]/.02*4*ulp).ravel()
 for off in [15860,17684]:tighten[off:off+1824]=(dt[:,None]*32*ulp).ravel()
 env=envelope(C,dt,lo,hi,ulp);prior=np.r_[raw-tighten,env['tightened'].ravel()];rawall=np.r_[raw,env['raw'].ravel()];frozen=np.load(D/'lattice_path_430/final.npz');scale=frozen['scale'];keep=frozen['keep'].astype(bool)
 V=np.c_[X[:,2:4],X[:,21:23],X[:,16:21]].ravel();shape=ini['shape_free'];oldbound=ini['bounds'][:1026][shape]
 coordinate=np.r_[V[shape]-oldbound[:,0],oldbound[:,1]-V[shape]];buffered=np.r_[prior[keep]/scale[keep]-BUFFER,coordinate-BUFFER]
 return dict(raw=rawall,prior=prior,buffered=buffered,X=X,V=V,coordinate_bounds=coordinate,dt=dt,C=C,window_raw=env['raw'],window_prior=env['tightened'],phase=f,orientation=p,source_slot_weights=W,gaps=gap,hover=np.array([(gap[99]-.012)/.012]))
def main():
 launch=read(D/'contact_lift_launch_440.json')
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 assert read(D/'contact_lift_preparation_audit_439/independent_audit.json')['passed'];O.mkdir(exist_ok=False);s=setup();w=np.load(D/'contact_lift_preparation_438/fixture.npz')['weights'];rows=[];rs=[];cc=[];tt=[];coms=[];eqs=[];start=time.monotonic()
 for i,(t,v,kind,index) in enumerate(s['queries']):
  r=record(s,w,t,v,kind,index);r,moment=endpoint_forces(s,r);r,root_info=solve_lift(s,r);rec,c,tau,com,eq=assess(s,r);r['measured_gap']=rec['pelvis_gap_m'];residual,*_=kinematics(s,r)
  rec.update(sample_kind=kind,sample_index=index,exact_time=r['exact_time'],phase=r['phase'],orientation_phase=r['orientation_phase'],guarded_gap=r['guarded_gap'],root_solve=root_info,endpoint_moment_root=moment,kinematic_residual=float(abs(residual).max()),new_nominal_left_support=r['stance']==['left'])
  rows.append(rec);rs.append(r);cc.append(c);tt.append(tau);coms.append(com);eqs.append(eq)
  # Durable completed-record evidence precedes all aggregate assertions. A later
  # metadata failure must never cause a completed root/FK loop to be repeated.
  with (O/'completed_records.jsonl').open('a') as journal:
   journal.write(json.dumps(dict(record=rec,q=r['q'].tolist(),root=np.r_[r['root'],Rotation.from_matrix(r['R']).as_quat()].tolist(),weights=r['weights'].tolist(),points=r['points'].tolist(),foot_frames=r['actual_foot_reference'].tolist(),com_target=r['com_reference'].tolist(),command=c.tolist(),torque=tau.tolist(),com=com.tolist(),equilibrium=eq.tolist(),normal=r['normal']))+'\n')
  if (i+1)%100==0:print(json.dumps(dict(records=i+1,total=1467,elapsed_s=time.monotonic()-start)),flush=True)
 cc=np.array(cc);tt=np.array(tt);coms=np.array(coms);eqs=np.array(eqs)
 (O/'records.json').write_text(json.dumps(rows,indent=2))
 np.savez_compressed(O/'constructed_states.npz',q=np.array([r['q'] for r in rs]),root=np.array([np.r_[r['root'],Rotation.from_matrix(r['R']).as_quat()] for r in rs]),weights=np.array([r['weights'] for r in rs]),points=np.array([r['points'] for r in rs]),foot_frames=np.array([r['actual_foot_reference'] for r in rs]),com_target=np.array([r['com_reference'] for r in rs]),normal=np.array([r['normal'] for r in rs]),command=cc,torque=tt,com=coms,equilibrium=eqs)
 old=old_rows(s,rs,cc,tt,coms)
 C=cc[:676][:,s['moving']];dc=np.diff(C,axis=0);ini=s['initial'];rates=ini['rates'];ulp=ini['ulp'];targets=10*C[1:]-9*C[:-1]
 clock_raw=np.c_[.02*rates-dc,.02*rates+dc,ini['command_hi']-targets,targets-ini['command_lo']]
 clock_prior=clock_raw-np.r_[4*ulp,4*ulp,32*ulp,32*ulp];clock_buffer=clock_prior/np.r_[.02*rates,.02*rates,np.full(32,.03)]-BUFFER
 # Half-clock checks concern the NEW implicit lift, not the old invalid434 slots.
 order=np.argsort([float(t) for t,_,kind,_ in s['queries'][:1351]],kind='stable');dense_f=np.array([rs[i]['phase'] for i in order]);dense_gap=np.array([rs[i]['measured_gap'] for i in order]);dense_t=np.array([rs[i]['time_s'] for i in order]);normal=(dense_t[:-1]>=float(s['times'][100]))&(dense_t[1:]<=float(s['times'][105]));normal_rows=.006*np.diff(dense_t)[normal]-(dense_gap[:-1]-dense_gap[1:])[normal]
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,construction=__import__('low_seat_contact_lift_438').__doc__,records=1467,clock_records=676,half_tick_records=675,vertex_records=116,new_nominal_nodes=read(D/'contact_lift_preparation_438/preparation.json')['changed_nominal_nodes'],exact_event_budgets=BUDGETS,exact_total_time_numerator=27,exact_total_time_denominator=2,source430_unchanged_prior_gates_passed=True,source430_unchanged_extra_buffer_passed=False,sampled_geometry_failures=[i for i,r in enumerate(rows) if not r['sampled_geometry_passed']],minimum_housing_m=min(r['clearance_m'] for r in rows),minimum_support_m=min(r['foot_cop_margin_m'] for r in rows),maximum_kinematic_residual=max(r['kinematic_residual'] for r in rows),maximum_equilibrium_residual=float(abs(eqs).max()),maximum_declared_frame_position_error_m=max(r['interpolated_foot_position_defect_m'] for r in rows),maximum_declared_frame_orientation_error_rad=max(r['interpolated_foot_orientation_defect_rad'] for r in rows),minimum_command_margin_rad=min(r['minimum_command_margin_rad'] for r in rows),maximum_effort_utilization=max(r['effort_utilization'] for r in rows),negative_corner_records=[i for i,r in enumerate(rows) if r['minimum_corner_fraction']<0],minimum_new_clock_rate_rad=float(clock_raw[:,:32].min()),minimum_new_clock_target_headroom_rad=float(clock_raw[:,32:].min()),minimum_prior_clock_reserve=float(clock_prior.min()),minimum_buffered_clock_row=float(clock_buffer.min()),original_node_constraint_min=float(old['raw'].min()),prior_node_constraint_min=float(old['prior'].min()),buffered_node_constraint_min=float(old['buffered'].min()),buffered_node_negative_rows=int((old['buffered']<0).sum()),hover_residual=float(abs(old['hover']).max()),minimum_sampled_descent_increment=float(np.diff(dense_f).min()),minimum_sampled_normal_residual_m=float(normal_rows.min()),exact_initial_command=bool(np.array_equal(cc[0],s['command0'])),fixed_q_exact=bool(all(np.array_equal(r['q'][s['fixed']],s['q0'][s['fixed']]) for r in rs)),fixed_commands_exact=bool(np.array_equal(cc[:,s['fixed']],np.tile(s['command0'][s['fixed']],(len(cc),1)))),root_iteration_cap=10,moment_iteration_cap=30,optimizer_calls=0,actual_forward_filter_ticks=0,executed_command_ticks=0,physics_ticks=0,policy_updates=0,continuous_clearance_certified=False,command_between_samples_certified=False,physical_admission=False)
 (O/'results.json').write_text(json.dumps(result,indent=2));(O/'records.json').write_text(json.dumps(rows,indent=2));np.savez_compressed(O/'states.npz',q=np.array([r['q'] for r in rs]),root=np.array([np.r_[r['root'],Rotation.from_matrix(r['R']).as_quat()] for r in rs]),weights=np.array([r['weights'] for r in rs]),points=np.array([r['points'] for r in rs]),foot_frames=np.array([r['actual_foot_reference'] for r in rs]),com_target=np.array([r['com_reference'] for r in rs]),normal=np.array([r['normal'] for r in rs]),command=cc,torque=tt,com=coms,equilibrium=eqs,clock_raw=clock_raw,clock_prior=clock_prior,clock_buffer=clock_buffer,normal_rows=normal_rows,**{f'node_{k}':v for k,v in old.items()})
 (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
