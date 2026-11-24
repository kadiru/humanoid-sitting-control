"""Retarget the observed pelvis marker onto the interior before lowering, with both soles anchored."""
from contact_corridor_path import *
from scipy.stats import qmc
def make():
    p=P/'corner_traction_16/evaluation';z=np.load(p/'trajectory.npz');m=read(p/'results.json');j=14
    assert sha(p/'trajectory.npz')==read(p/'independent_audit.json')['files']['trajectory.npz']
    B=z['body'][:,j].astype(float);B[:,:,:3]-=m['origins'][j]
    q=z['q'][:,j].astype(float);cmd=z['command'][:,j,:23].astype(float)
    ft=[m['body_names'].index(n) for n in feet_names];seat=Seat(geoms[21])
    marker=np.array([-.001,0.,-.152]);rng=qmc.Sobol(5,scramble=True,seed=472);samples=np.r_[np.zeros((1,5)),2*rng.random_base2(9)-1]
    results=[]
    for end in [342,355,370,385]:
        b=B[end];R0=Rot.from_quat(b[0,3:]).as_matrix();root0=b[0,:3]
        angles=samples[:,2:]*[.15,.145,.135];Rs=Rot.from_rotvec(angles).as_matrix()@R0
        roots=root0+np.c_[samples[:,:2]*[.018,.022],np.zeros(len(samples))]
        cps=(Rs@marker)+roots;h,ma=lookup(seat,cps);ok=np.isfinite(h)&(ma>=.018)
        roots[:,2]+=np.where(ok,h-cps[:,2]-.0005,0.)
        qs=np.tile(q[end],(len(samples),1));errs=np.zeros(len(samples));F=b[ft]
        from planted_edge_fixed_search import batched_ik
        for chain,ids,foot in zip(chains,legs,F):
            qs[:,ids],d,a=batched_ik(chain,qs[:,ids],((foot[:3]-roots)[:,None]@Rs)[:,0],Rs.transpose(0,2,1)@Rot.from_quat(foot[3:]).as_matrix(),ids);errs=np.maximum(errs,d+.12*a)
        for k in np.flatnonzero(ok&(errs<.0008)):
            st=robot.state(qs[k],Rs[k],roots[k]);gaps=measure(st,seat);_,cm=ground(st,F)
            if gaps[2]<.001:continue
            # Loaded native contacts on successful chairs lie 1-5 mm above the vertex proxy.
            # Check both proxy order and the orientation of the local marker.
            clear=gaps[2];rim=gaps[1]-gaps[0];qoff=abs(qs[k,:23]-q[end,:23]).max()
            tau,_=ground(st,F);orig,_=ground(robot.state(q[end],R0,root0),F)
            target=cmd[end]+qs[k,:23]-q[end,:23]+(tau-orig)/kp
            drive=np.minimum(target-lo[:23],hi[:23]-target).min()
            if drive<0:continue
            cost=min(.017,clear,.003+rim+.002,.14*drive)-.011*qoff-.10*np.linalg.norm(roots[k,:2]-root0[:2])
            results.append((cost,dict(end=end,offset=(roots[k]-root0).tolist(),rotation=angles[k].tolist(),housing=clear,rim_relative=rim,margin=ma[k],command_margin=drive)))
    results.sort(key=lambda x:-x[0]); keep=[]
    for _,r in results:
        if all(r['end']!=s['end'] or np.linalg.norm(np.array(r['rotation'])-s['rotation'])>.09 for s in keep):keep.append(r)
        if len(keep)==6:break
    # An unrotated over-rim path isolates the effect of height from pose correction.
    for end in (345,365):
        r=B[end,0];cp=Rot.from_quat(r[3:]).as_matrix()@marker+r[:3];h,_=lookup(seat,cp)
        keep.append(dict(end=end,offset=[0,0,float(h-cp[2]-.0005)],rotation=[0,0,0]))
    assert len(keep)>=4
    paths=[];gains=[];audit=[]
    for spec in keep:
        end=spec['end'];start=188; rr=np.array(spec['rotation']);delta=np.array(spec['offset'])
        ts=np.unique(np.r_[np.arange(start,end+54,4),end,end+53]);deltaq=[];history=[];jac=[]
        for t in ts:
            k=min(end,t);base=B[k,0,:3];R0=Rot.from_quat(B[k,0,3:]).as_matrix();foot=B[k,ft]
            w=smooth((t-start)/(end-20-start));wz=smooth((t-start)/(end-start))
            R=Rot.from_rotvec(rr*w).as_matrix()@R0;root=base+delta*w
            root[2]=base[2]+delta[2]*wz+.009*smooth((t-start)/95)*(1-smooth((t-end+6)/40))-.002*smooth((t-end-6)/40)
            sol,err=fit(q[k],root,R,foot);st=robot.state(sol,R,root);tau,_=ground(st,foot);oldtau,_=ground(robot.state(q[k],R0,base),foot)
            deltaq.append((sol[:23]-q[k,:23])+(tau-oldtau)/kp);history.append([*measure(st,seat),err])
        end2=int(ts[-1]);tt=np.arange(1000);phase=np.minimum(tt,end);corr=PchipInterpolator(ts,np.array(deltaq),axis=0)(tt.clip(ts[0],ts[-1]))
        corr[:start]=0;target=cmd[phase]+corr;ids=legs.ravel()
        step=np.max(abs(np.diff(target[:,ids],axis=0)),1);clk=np.r_[0,np.cumsum(np.where(tt[1:]<start,1.,np.maximum(1.,step/.00288)))]
        path=np.stack([np.interp(tt,clk,col) for col in target.T],1);prev=np.vstack((path[0],path[:-1]));req=prev+10*(path-prev)
        ready=int(np.ceil(clk[end2]));assert ready<740
        paths.append(req);matrix=np.load(P/'measured_corner_path_30.npz')['motion'][end,84:].reshape(12,6);gains.append(matrix)
        audit.append(dict(spec=spec,ready=ready,swept=np.min(history,axis=0).tolist(),max_ik=max(k[-1] for k in history)))
    write(P/'height_bridge_plan.json',audit);np.savez_compressed(P/'height_bridge_plan.npz',commands=paths,J=gains)
    print([(r['spec']['end'],r['ready'],np.round(r['swept'][:3],4).tolist()) for r in audit],flush=True)
if __name__=='__main__':make()

