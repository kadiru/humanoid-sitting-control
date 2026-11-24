"""Build a causal pelvis-marker target from a measured seat-rim touch and the interior surface."""
from contact_corridor_path import P,read,write,Rot,Seat,geoms,lookup,smooth
import numpy as np
def build():
    run=P/'refined_native_trials/evaluation';z=np.load(run/'trajectory.npz');m=read(run/'results.json')
    w=np.load(run/'contact_load_geometry.npz')['observations'][25:];logs=np.load(run/'corridor_feedback.npz')['data']
    corridors=np.load(P/'contact_corridor_path_6.npz')['motion']
    bank=np.load(P/'refined_native_pivot.npz');spec=read(P/'refined_native_pivot_specs.json')
    seat=Seat(geoms[6]);J=[];commands=[];xyz=[];caps=[];attrs=[]
    for case in (2,8):
        first=int(np.flatnonzero(w[:,case,0,9]==1)[0])
        cp=w[first,case,0,:3];b=z['body'][first,case,0].astype(float);b[:3]-=m['origins'][case]
        rr=Rot.from_quat(b[3:]).as_matrix();skin=rr.T@(cp-b[:3])
        h=logs[:,case];k=h[:,1].astype(int);ref=corridors[k];ori=Rot.from_rotvec(-h[:,8:11]).as_matrix()@ref[:,72:81].reshape(-1,3,3)
        marker=ref[:,69:72]-h[:,5:8]+ori@skin
        basis=ori[first-40]@rr.T
        # Use a patch within the eroded legal top and continue to check every contact in the evaluator.
        dst=np.array(spec[case]['interior']);dst[2]+=.0018
        assert seat.lookup(dst[None])[0][0]
        for aim in (dst, dst+np.array([-.007,-.005,0.])):
            height,margin=lookup(seat,aim);aim=aim.copy();aim[2]=height
            if margin<.012:continue
            shift=basis@(aim-cp);tick=np.arange(1000);cut=first-6
            a=bank['commands'][case].copy();a[cut:]=np.asarray(z['command'][cut,case,:23])
            old=marker[np.clip(tick-40,0,cut-40)].copy()
            for lift,gain,integral in [(.009,.9,.5),(.014,1.35,1.2),(.019,1.65,1.),(.012,1.85,1.6)]:
                goal=old.copy()
                dis=smooth((tick-475)/118)
                goal[:,:2]+=dis[:,None]*shift[:2]
                goal[:,2]+=dis*lift
                commands.append(a);J.append(bank['J'][case]);xyz.append(goal)
                cap=goal[-1,2]-lift+shift[2]-.003;caps.append(cap)
                attrs.append(dict(case=case,skin=skin.tolist(),gain=gain,integral=integral,source_xyz=cp.tolist(),target_xyz=aim.tolist(),lift=lift,cut=cut))
    np.savez_compressed(P/'pelvis_marker_feedback.npz',commands=commands,J=J,goals=xyz,caps=caps)
    rows=[dict(asset_id=6,config=dict(trajectory='pelvis_marker_feedback.npz',start=440,trigger=.008,load=.09,variant=i,marker=v)) for i,v in enumerate(attrs)]
    write(P/'marker_feedback_trials.json',dict(routes={'6':rows[0]['config']},diagnostic_rows=rows))
    print(len(rows),[r['source_xyz'] for r in attrs[::8]])
if __name__=='__main__':build()

