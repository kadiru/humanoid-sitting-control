"""Plan from a measured two-foot hover using native pelvis contact points from the 25 passes."""
from contact_corridor_path import *
from planted_edge_fixed_search import batched_ik
from scipy.stats import qmc
import csv
def fit_skin():
    skin=[];records=list(csv.DictReader((P/'corner_traction_cases.csv').open()))
    for h in (0,16):
        p=P/f'corner_traction_{h}/evaluation';z=np.load(p/'trajectory.npz');m=read(p/'results.json');B=z['body']
        w=np.load(p/'contact_load_geometry.npz')['observations'][25:]
        for j in range(16):
            if records[h+j]['supported_sit']!='True':continue
            idx=np.flatnonzero(w[:,j,0,8]==1)[::13]
            if len(idx):
                b=B[idx,j,0].astype(float);b[:,:3]-=m['origins'][j]
                skin.extend(((w[idx,j,0,:3]-b[:,:3])[:,None]@Rot.from_quat(b[:,3:]).as_matrix())[:,0])
    pts=thin(np.array(skin),.002)
    return np.vstack((pts,pts*np.array([1.,-1.,1.])))
def build():
    ev=P/'intercept_contact_trials/evaluation';z=np.load(ev/'trajectory.npz');meta=read(ev/'results.json');j=6;end=587
    assert sha(ev/'trajectory.npz')==read(ev/'independent_audit.json')['files']['trajectory.npz']
    b=z['body'][:,j].astype(float);b[:,:,:3]-=meta['origins'][j];q=z['q'][:,j].astype(float);cmd=z['command'][:,j,:23].astype(float)
    idx=[meta['body_names'].index(n) for n in feet_names];feet=b[end,idx];ref=b[end,0,:3];R0=Rot.from_quat(b[end,0,3:]).as_matrix();marker=fit_skin();s=Seat(geoms[6])
    options=[];rng=np.random.default_rng(23921);seeds=np.zeros((1,5))
    for gen in range(7):
        a=seeds[rng.integers(len(seeds),size=460)]+rng.normal(size=(460,5))*np.array([.017,.021,.08,.10,.13])*(.81**gen)
        a[:len(seeds)]=seeds
        a=np.clip(a,[-.04,-.052,-.19,-.23,-.31],[.035,.052,.19,.23,.31])
        R=Rot.from_rotvec(a[:,2:]).as_matrix()@R0;rt=np.tile(ref,(len(a),1));rt[:,:2]+=a[:,:2];rt[:,2]=0.
        pv=marker@R.transpose(0,2,1)+rt[:,None];h,ma=lookup(s,pv)
        height=np.where(ma>=.016,h-pv[:,:,2],-np.inf).max(1);use=np.isfinite(height)&(abs(height-ref[2])<.034)
        rt[:,2]=np.where(use,height-.0008,ref[2])
        qs=np.tile(q[end],(len(a),1));err=np.zeros(len(a))
        for ids,c,foot in zip(legs,chains,feet):
            qs[:,ids],ep,ea=batched_ik(c,qs[:,ids],((foot[:3]-rt)[:,None]@R)[:,0],R.transpose(0,2,1)@Rot.from_quat(foot[3:]).as_matrix(),ids);err=np.maximum(err,ep+.12*ea)
        for k in np.flatnonzero(use&(err<.0008)):
            st=robot.state(qs[k],R[k],rt[k]);gap=measure(st,s)
            rim=rt[k,2]-np.where(ma[k]<.012,h[k]-pv[k,:,2],-np.inf).max()
            tau,cm=ground(st,feet);old,_=ground(robot.state(q[end],R0,ref),feet)
            target=cmd[end]+qs[k,:23]-q[end,:23]+(tau-old)/kp
            margin=np.minimum(target-lo[:23],hi[:23]-target).min()
            tilt=np.arccos(R[k,2,2])
            score=min(.011,min(.15*margin,gap[2]-.003,rim-.002))-.035*np.linalg.norm(a[k,:2])-.006*np.linalg.norm(a[k,2:])-.18*max(0.,tilt-.655)
            options.append((float(score),dict(offset=(rt[k]-ref).tolist(),rotation=a[k,2:].tolist(),rim=float(min(.30,rim)),other=float(gap[2]),drive=float(margin)),a[k].copy()))
        options.sort(key=lambda r:-r[0]);seeds=np.stack([r[2] for r in options[:12]])
    selected=[]
    for _,r,a in options:
        if all(np.linalg.norm(a-np.r_[s0['offset'][:2],s0['rotation']])>.037 for s0 in selected):selected.append(r)
        if len(selected)>=6:break
    # Include the actual hovering posture with a vertical load transfer as a control.
    selected.extend([dict(offset=[-.003,0,-.004],rotation=[.015,0,0]),dict(offset=[-.004,-.003,-.004],rotation=[.022,-.012,0])])
    duration=75;start=end-121;paths=[];gains=[]
    for r in selected:
        ts=np.unique(np.r_[np.arange(start,end+duration,3),end,end+duration])
        corr=[];clear=[]
        for t in ts:
            k=min(t,end);ori=Rot.from_quat(b[k,0,3:]).as_matrix();root=b[k,0,:3].copy()
            foot=b[k,idx];w=smooth((t-start)/(end-start+12));f=smooth((t-start)/(end-start+46))
            R=Rot.from_rotvec(w*np.array(r['rotation'])).as_matrix()@ori
            delta=np.array(r['offset']);root[:2]+=w*delta[:2];root[2]+=f*delta[2]+.005*w*(1-smooth((t-end+4)/57))
            ang,e=fit(q[k],root,R,foot);st=robot.state(ang,R,root);tau,_=ground(st,foot);old,_=ground(robot.state(q[k],ori,b[k,0,:3]),foot)
            corr.append(ang[:23]-q[k,:23]+(tau-old)/kp);clear.append(measure(st,s))
        tt=np.arange(1000);cc=PchipInterpolator(ts,corr,axis=0)(tt.clip(ts[0],ts[-1]));cc[:start]=0;target=cmd[np.minimum(tt,end)]+cc;ids=legs.ravel()
        d=np.max(abs(np.diff(target[:,ids],axis=0)),1);clock=np.r_[0,np.cumsum(np.where(tt[1:]<start,1.,np.maximum(1.,d/.00288)))]
        actual=np.stack([np.interp(tt,clock,col) for col in target.T],1);prev=np.vstack((actual[0],actual[:-1]));paths.append(prev+10*(actual-prev))
        gains.append(np.load(P/'contact_corridor_path_6.npz')['motion'][575,84:].reshape(12,6));r['ready']=int(np.ceil(clock[ts[-1]]));r['swept']=np.min(clear,0).tolist()
    write(P/'native_skin_bridge.json',selected);np.savez_compressed(P/'native_skin_bridge.npz',commands=paths,J=gains,skin=marker)
    print([(r['offset'],r['rotation'],r.get('other'),r.get('rim'),r['ready']) for r in selected],flush=True)
if __name__=='__main__':build()

