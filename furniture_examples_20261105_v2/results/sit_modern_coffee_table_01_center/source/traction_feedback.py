"""Sensor-only touchdown balance correction; no simulator labels are read."""
import torch
def hip_touchdown(env,ref,ramp_seconds):
    if not env.planning_enabled:return ref
    elapsed=(env.episode_length_buf-env.near_start).float()*.02
    u=(elapsed/5*24).clamp(0,24);lo=u.floor().long().clamp_max(23);f=u-lo;ids=torch.arange(env.num_envs,device=env.device)
    target=(1-f[:,None])*env.target_paths[ids,lo]+f[:,None]*env.target_paths[ids,lo+1]
    velocity=(env.target_paths[ids,lo+1,1]-env.target_paths[ids,lo,1])*24/5*(u<24)
    g=env.robot.data.projected_gravity_b.torch;pitch=torch.atan2(g[:,0],-g[:,2])
    correction=.6*(pitch-target[:,1])+.15*(env.robot.data.root_ang_vel_b.torch[:,1]-velocity)
    if env.estimated_root_x is not None:correction+=env.estimated_root_x-target[:,0]
    correction=correction.clamp(-.25,.25)*(elapsed/.2).clamp(0,1)*env.target_ready*(env.near_start>=0)
    old=((elapsed-.5)/1).clamp(0,1);old=old.square()*(3-2*old)
    new=(elapsed/ramp_seconds).clamp(0,1);new=new.square()*(3-2*new)
    delta=(new-old)*env.target_ready*(env.near_start>=0)
    ref[:,env.hip_pitch_ids]+=correction[:,None]*delta[:,None]
    ref[:,env.ankle_ids]-=correction[:,None]*delta[:,None]
    ref[:,env.ankle_ids]+=(-3*env.foot_pitch).clamp(-.15,.15)*delta[:,None]
    return ref
