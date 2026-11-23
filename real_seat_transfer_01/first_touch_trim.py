"""Use the measured pre-impact state to find a nearby clean pelvis touchdown with fixed soles."""
from contact_corridor_path import *
from planted_edge_fixed_search import batched_ik
from scipy.stats import qmc
from scipy.ndimage import gaussian_filter1d

def build_trim(i):
    folder=P/f'sole_moment_{i//16*16}/evaluation';audit=read(folder/'independent_audit.json')
    assert sha(folder/'trajectory.npz')==audit['files']['trajectory.npz']
    z=np.load(folder/'trajectory.npz');meta=read(folder/'results.json');j=i%16
    t=int(np.flatnonzero(z['flags'][:,j].any(1))[0]);e=t-1;body=z['body'][:,j].astype(float);body[:,:,:3]-=meta['origins'][j];baseq=z['q'][:,j].astype(float)
    shidx=[meta['body_names'].index(n) for n in feet_names]
    b=body[e];R0=Rot.from_quat(b[0,3:]).as_matrix();p0=b[0,:3];feet=b[shidx];s=Seat(geoms[i]);q0=baseq[e]
    loaded=z['physical_force_w'][e,j,5:,2]/z['physical_bodyweight_N'][e,j];best=None
    samples=qmc.Sobol(5,scramble=True,seed=233+i).random_base2(15)
    for raw in np.array_split(samples,160):
        N=len(raw);a=(2*raw-1)*[.028,.031,.102,.135,.145]
        R=Rot.from_rotvec(a[:,2:]).as_matrix()@R0;root=np.tile(p0,(N,1));root[:,:2]+=a[:,:2];root[:,2]=0
        v=verts['pelvis']@R.transpose(0,2,1)+root[:,None];h,ma=lookup(s,v);lift=h-v[:,:,2]
        good=np.where(ma>=.016,lift,-np.inf);height=good.max(1);rim=height-np.where(ma<.012,lift,-np.inf).max(1);root[:,2]=np.where(np.isfinite(height),height,1)
        use=np.isfinite(height)&(abs(root[:,2]-p0[2])<.025)&(rim>.001)
        k=np.flatnonzero(use)
        if not len(k):continue
        rot=R[k];rt=root[k];pv=v[k];rim=rim[k];qs=np.tile(q0,(len(k),1));err=np.zeros(len(k))
        cp=pv[np.arange(len(k)),good[k].argmax(1)];cp[:,2]+=rt[:,2]
        for c,ids,foot in zip(chains,legs,feet):
            sol,ep,ea=batched_ik(c,qs[:,ids],((foot[:3]-rt)[:,None]@rot)[:,0],rot.transpose(0,2,1)@Rot.from_quat(foot[3:]).as_matrix(),ids);qs[:,ids]=sol;err=np.maximum(err,ep+.12*ea)
        for z1 in np.flatnonzero(err<.001):
            st=robot.state(qs[z1],rot[z1],rt[z1]);gaps=measure(st,s);cm=np.average(st[1],0,weights=robot.mass)
            poly=[]
            for name,foot in zip(feet_names,feet):
                v=model.vertices[name]@Rot.from_quat(foot[3:]).as_matrix().T+foot[:3];poly.extend(v[v[:,2]<v[:,2].min()+.002, :2])
            eq=ConvexHull(poly).equations;cop=(cm[:2]-.17*cp[z1,:2])/.83;bal=-max((eq[:,:2]@cm[:2]+eq[:,2]).max(),(eq[:,:2]@cop+eq[:,2]).max())
            if bal<.002 or gaps[2]<.002:continue
            tau,_=ground(st,feet);cmd=qs[z1,:23]+tau/kp;drive=np.minimum(cmd-lo[:23],hi[:23]-cmd).min()
            if drive<0:continue
            score=min(.012,rim[z1],gaps[2]-.001,bal)-.014*np.linalg.norm(qs[z1,:23]-q0[:23])
            if best is None or score>best[0]:
                best=(score,dict(root=rt[z1].tolist(),R=rot[z1].tolist(),q=qs[z1].tolist(),cp=cp[z1].tolist(),rim_m=float(rim[z1]),other_gap_m=float(gaps[2]),pressure_m=float(bal),joint_margin_rad=float(drive),event=t))
    if best is None:
        write(P/f'first_touch_trim_{i}.json',dict(admitted=False,event=t));print(i,'no candidate',flush=True);return
    spec=best[1];target=np.array(spec['root']);rot=np.array(spec['R']);offset=target-p0;turn=Rot.from_matrix(rot@R0.T).as_rotvec()
    delta=[];last_p=[];checked=[]
    for tt in np.arange(max(40,e-145),e+1,2).tolist()+[e,e+14,e+38,e+68]:
        tt=int(tt);k0=min(tt,e);qref=baseq[k0];base=body[k0,0,:3];R=Rot.from_quat(body[k0,0,3:]).as_matrix();foot=body[k0,shidx]
        weight=smooth((tt-e+143)/110);R1=Rot.from_rotvec(turn*weight).as_matrix()@R
        p=base+offset*weight;p[2]+=.005*weight*(1-smooth((tt-e+17)/48))-.0015*smooth((tt-e-13)/54)
        sol,err=fit(qref,p,R1,foot);st=robot.state(sol,R1,p);old=robot.state(qref,R,base)
        tau,_=ground(st,foot);oldtau,_=ground(old,foot);corr=sol[:23]-qref[:23]+(tau-oldtau)/kp
        dd=measure(st,s);delta.append((tt,corr));last_p.append(p);checked.append(dd)
    tt,inds=np.unique([d[0] for d in delta],return_index=True);cc=np.stack([delta[k][1] for k in inds])
    dense=PchipInterpolator(tt,cc,axis=0)(np.arange(1000).clip(tt[0],tt[-1]))
    spec.update(admitted=True,offset=offset.tolist(),rotation=turn.tolist(),correction_peak=float(abs(dense).max()),path_min=np.min(checked,0).tolist())
    write(P/f'first_touch_trim_{i}.json',spec);np.savez_compressed(P/f'first_touch_trim_{i}.npz',delta=dense.astype('f'))
    print(i,[(k,spec[k]) for k in ('rim_m','other_gap_m','offset','path_min')],flush=True)
if __name__=='__main__':build_trim(int(sys.argv[1]))
