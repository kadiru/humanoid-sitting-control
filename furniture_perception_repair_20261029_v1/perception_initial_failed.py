"""Depth-only, task-directed support selection. No asset identity or truth geometry."""
import numpy as np
from humanoid_sitting.shared_frame import apply
from humanoid_sitting.supported_surface import fit_observed_planes

def cloud(depth,K,T):
    d=np.asarray(depth).squeeze();h,w=d.shape;v,u=np.mgrid[:h,:w]
    valid=np.isfinite(d)&(d>.08)&(d<3.);z=np.where(valid,d,0.)
    # RTX samples the center of each pixel; K's principal point is in pixel-edge coordinates.
    C=np.stack(((u+.5-K[0,2])*z/K[0,0],(v+.5-K[1,2])*z/K[1,1],z),-1)
    return apply(T,C.reshape(-1,3)).reshape(h,w,3),valid

def observe(model,K,T,depth,q,gravity,intent,prior=None):
    assert intent in ['seat','arm','back','top']
    P,valid=cloud(depth,K,T);h,w=valid.shape;poses=model.fk(q)
    valid &=model.self_mask(P.reshape(-1,3),poses,margin=.009).reshape(h,w)
    dx=P[1:-1,2:]-P[1:-1,:-2];dy=P[2:,1:-1]-P[:-2,1:-1]
    n=np.cross(dx,dy);length=np.linalg.norm(n,axis=-1)
    ok=valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]
    ok &=(length>1e-8)&(np.linalg.norm(dx,axis=-1)<.15)&(np.linalg.norm(dy,axis=-1)<.15)
    pts=P[1:-1,1:-1][ok];norm=n[ok]/length[ok,None]
    norm[(norm*pts).sum(1)>0]*=-1
    up=-np.asarray(gravity);up/=np.linalg.norm(up)
    # Body-forward direction projected into the gravity-horizontal plane.
    forward=np.array([1.,0.,0.]);forward-=up*np.dot(forward,up);forward/=np.linalg.norm(forward)
    compatible=(abs(norm@up)<.65)&(norm@forward>.5) if intent=='back' else norm@up>.75
    # Exclude incompatible faces before clustering: a large back must not consume the seat's clusters.
    planes=fit_observed_planes(pts[compatible],norm[compatible],min_points=20)
    soles=np.concatenate([apply(poses[s+'_ankle_roll_link'],model.vertices[s+'_ankle_roll_link']) for s in ['left','right']]);floor=float(np.min(soles@up))
    candidates=[p for p in planes if p['center'][0]<.1 and np.linalg.norm(p['center'])<1.5 and p['area']>.006
        and not(abs(p['normal']@up)>.97 and abs(p['center']@up-floor)<.055)]
    info=dict(intent=intent,planes=len(planes),candidate_count=len(candidates),actor_truth_input=False,candidates=[])
    for p in candidates:
        info['candidates'].append(dict(normal=p['normal'].tolist(),center=p['center'].tolist(),offset=p['offset'],span=p['span'].tolist(),points=len(p['points']),rms=p['rms']))
    if not candidates:return None,info
    if prior is not None and prior[11] and prior[12]<.5:
        matched=[p for p in candidates if p['normal']@prior[2:5]>.94 and abs(p['offset']-prior[5])<.08]
        if matched:candidates=matched
        else:info['association_rejected']=True;return None,info
    if intent in ['arm','back']:
        # Closest compatible face across the robot's rear; no prescribed height or asset label.
        p=max(candidates,key=lambda p:float(p['center']@forward))
    else:p=max(candidates,key=lambda p:p['area'])
    pelvis=float((model.vertices['pelvis']@p['normal']+p['offset']).min())
    rear=np.concatenate([apply(poses[b],model.vertices[b]) for b in ['torso_link','logo_link']]);torso=float((rear@p['normal']+p['offset']).min())
    info.update(selected_normal=p['normal'].tolist(),selected_center=p['center'].tolist(),rms=p['rms'])
    return np.array([pelvis,torso,*p['normal'],p['offset'],*p['center'],*p['span'],1.,0.],np.float32),info
