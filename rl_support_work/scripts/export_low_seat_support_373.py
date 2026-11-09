"""Complete support/COM path output with mandatory original geometry and physical criteria."""
import copy,json,datetime
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply,transform
def export_result(p,solution,opt,preparation,hashes,O,slack,scale,keep,hard):
 N=p['N']
 NP=p['NP']
 last=p['last']
 calc=p['calc']
 evaluate=p['evaluate']
 frozen=p['frozen']
 basecmd=p['basecmd']
 change=p['change']
 moving=p['moving']
 oldnodes=p['oldnodes']
 seed_nodes=p['seed_nodes']
 sides=p['sides']
 metrics=p['metrics']
 m=p['m']
 n=p['n']
 d=p['d']
 limits=p['limits']
 names=p['names']
 effort=p['effort']
 prior=p['prior']
 pose_change=p['pose_change']
 eq,iq=evaluate(solution);XX=last['X'];Y,Q,Ts,Cs=calc(XX,True);path=copy.deepcopy(frozen);nodes=[];allcmd=basecmd.copy();allcmd[change[:,None],np.array(moving)[None,:]]=Y[:,14:30]
 for i,k in enumerate(change):
  if k>=104:continue
  old=frozen[k];feet={s:(np.array(old['foot_targets'][s]['position']),np.array(old['foot_targets'][s]['rotation'])) for s in sides};a=metrics(Ts[i,:3,3],Ts[i,:3,:3],Q[i],feet,old['stance']);f,p=XX[i,2:4];a.update(phase=old['phase'],u=old['u'],swing_attitude_bound_rad=old['swing_attitude_bound_rad'],approach_fraction=float(f),orientation_fraction=float(p),com_target_xy=(np.array(old['com'][:2])+.1*XX[i,21:23]).tolist(),original_nominal_com_target_xy=old['com_target_xy'],com_target_error_m=float(np.linalg.norm(np.array(old['com'][:2])+.1*XX[i,21:23]-Cs[i,:2])),multiknot_constraint_residual=float(abs(Y[i,:14]).max()));path[k]=a
 for k,e in enumerate(path):
  old=oldnodes[k];v=seed_nodes[k];P=np.array(old['corner_world_points']);ss=old['corner_sides'];cnt=len(P);normal=v['assumed_seat_normal_fraction']
  if k:
   i=k-1;w=Y[i,46:46+cnt];tau=Y[i,30:46];T=transform(Rotation.from_quat(e['root'][3:]).as_matrix(),e['root'][:3]);pv=apply(T,m.vertices['pelvis']);pd=pv@n+d;seat=pv[pd<=pd.min()+1e-7].mean(0);seat-=n*(seat@n+d);sv=normal/n[2];com=np.array(e['com']);res=float(max(abs(w.sum()+sv-1),abs(w@P[:,:2]+sv*seat[:2]-com[:2]).max()))
  else:w=np.array(v['corner_fractions']);tau=np.array(v['planned_static_torque_Nm']);res=v['max_equilibrium_residual']
  cmd=allcmd[k];margin=float(min(min(cmd[j]-limits[names[j]][0],limits[names[j]][1]-cmd[j]) for j in moving));eu=float(max(abs(tau)/effort));nodes.append(dict(knot=k,command=cmd.tolist(),planned_static_torque_Nm=tau.tolist(),corner_fractions=w.tolist(),corner_world_points=P.tolist(),corner_sides=ss,assumed_seat_normal_fraction=normal,planned_right_load_fraction=float(sum(ww for ww,side in zip(w,ss) if side=='right')),min_command_margin_rad=margin,maximum_effort_utilization=eu,max_equilibrium_residual=res))
 failed=[k for k,e in enumerate(path) if not e['model_pose_passed']];duration=float(.5+solution[N*9:].sum());margin=min(v['min_command_margin_rad'] for v in nodes);util=max(v['maximum_effort_utilization'] for v in nodes);forceok=all(min(v['corner_fractions'])>=0 and v['max_equilibrium_residual']<1e-8 for v in nodes);constraints=bool(abs(eq).max()<=1e-8 and iq.min()>=-1e-8);swingheight=min(e['foot_min_z']['right'] for e in path if e['phase']=='swing' and .2<e['u']<.8)
 opt.update(objective_time_s=duration,objective_slack=float(slack),max_equality_residual=float(abs(eq).max()),minimum_inequality=float(iq.min()),numerical_constraints_passed=constraints,local_not_global=True)
 dense=dict(parameters=prior['dense']['parameters'],knots=115,path=path,command_nodes=nodes,segment_durations_s=solution[N*9:].tolist(),minimum_synthetic_quiet_s=.5,failed_knots=failed,min_clearance_m=min(e['clearance_m'] for e in path),min_cop_margin_m=min(e['foot_cop_margin_m'] for e in path),max_foot_position_error_m=max(e['foot_position_error_m'] for e in path),max_foot_orientation_error_rad=max(e['foot_orientation_error_rad'] for e in path),min_actual_swing_sole_height_m=swingheight,min_command_margin_rad=margin,maximum_effort_utilization=util,model_geometry_passed=bool(not failed and swingheight>=.005),model_command_passed=bool(margin>=.03 and util<=1 and forceok),guarded_continuous_timing_s=duration,guarded_timing_constraints_passed=bool(constraints and duration<=13.5),actual_filter_or_dynamic_check=False,physical_admission=False)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=dict(preparation,changed_pose_knots=pose_change.tolist()),endpoint=prior['endpoint'],stance_target_constraint=prior['stance_target_constraint'],dense=dense,unchanged_pose_knots=[0]+list(range(104,115)),optimizer=opt,source_input_sha256=hashes,physics_ticks=0,policy_optimizer_updates=0,physical_admission=False)
 result['optimizer']['objective_is_dimensionless_slack']=True
 result['optimizer']['objective_time_s_is_legacy_duration_field']=True
 result['optimizer']['scaled_original_violation']=float(max(0.,np.max(-iq[keep]/scale[keep]),np.max(abs(eq))))
 result['optimizer']['scaled_hard_violation']=float(max(0.,np.max(-iq[keep&hard]/scale[keep&hard]),np.max(abs(eq))))
 result['optimizer']['scaled_soft_violation']=float(max(0.,np.max(-iq[keep&~hard]/scale[keep&~hard])))
 result['optimizer']['slack_is_not_admission_tolerance']=True
 result['dense']['actual_filter_or_dynamic_check']=False
 np.savez_compressed(O/'final.npz',x=solution,full_coordinates=XX,eq=eq,ineq=iq,local_outputs=Y,slack=slack,scale=scale,keep=keep,hard=hard)
 (O/'results.json').write_text(json.dumps(result,indent=2))
 return result
