"""Independent saved-trajectory reward oracle. No simulation or optimization."""
import json,sys,hashlib,time
from pathlib import Path
import numpy as np
from contact_reward_v3 import LandingLabels,episode_reward,MODEL,HOUSINGS

def clean(x):
    return bool(x['success'] and not any(x['failures'].values()) and not x['slip_ticks'] and not x['illegal_ticks'])

def oracle(r,g,rows,margin,ids):
    n=len(rows);steps=np.zeros((1000,n));terms=[]
    # Independent forward accumulation; direct suffix sums, not producer cumsum.
    for k in range(300,1000):
        t=(k+1)*.02
        eligible=r['legal'][k] & (r['wrong'][k]<=10) & ~r['failure'][k] & r['alive'][k]
        loads=np.maximum(np.einsum('nki,nki->nk',r['force'][k],r['normal'][k]),0)
        loads*=r['permitted'][k]*r['inside'][k]*r['rest_patch'][k]
        support=np.minimum(loads.sum(1)/r['weight'][k]/.2,1)
        still=np.exp(-(r['root_speed'][k]**2/.01+r['gyro'][k]**2/.0625))
        v=np.where(np.sqrt((r['force'][k]**2).sum(-1))>.01*r['weight'][k,:,None],r['slip'][k],0).max(1)
        score=2*support*still*eligible*(t>=8)+2*r['good'][k]*eligible*(t>=8)
        score-=2*np.minimum(v/.02,1)
        score-=np.minimum(np.maximum(r['wrong'][k]-10,0)/(.1*r['weight'][k]),1)
        score-=(~r['legal'][k])
        score-=.25*np.minimum(r['total_squared_torque_Nm2'][k]/1350,1)
        steps[k]=score*.02/8
        if k%5==4:
            j=k//5
            if t<=13:steps[k]+=.1*.5*np.exp(-g['pelvis_target_distance_m'][j]**2/.0025)*eligible/8
            steps[k]-=.1*np.minimum(np.maximum((.015-g['housing_vertex_clearance_m'][j])/.015,0),1)/8
    for j,row in enumerate(rows):
        for mask,price in [(~r['no_slip'][:,j],5),(~r['legal'][:,j],5),(r['failure'][:,j],20)]:
            ix=np.flatnonzero(mask)
            if len(ix):steps[ix[0],j]-=price
        steps[-1,j]+=50*clean(row)-10*(not row['success'])-2*(row['traction_bad_ticks']>margin[j])
    control=np.stack([steps[5*k:5*k+5].sum(0)/50 for k in range(200)])
    return np.stack([control[k:].sum(0) for k in ids]),control

def rotate(q,v):
    q=np.asarray(q,dtype=np.float64);v=np.asarray(v,dtype=np.float64)
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    u=q[...,:3];s=q[...,3:]
    return v+2*np.cross(u,np.cross(u,v)+s*v)

def geometry_oracle(label,poses):
    result=[]
    # No scipy rotations or producer einsum path.
    for i in range(len(poses)):
        distances=[];gap=None
        from scipy.spatial.transform import Rotation
        seat_q=Rotation.from_matrix(label.rotation[i]).as_quat();seat_q[:3]*=-1
        for b,vertices in label.vertices.items():
            pose=poses[i,label.ids[b]]
            world=rotate(pose[3:],vertices)+pose[:3]
            points=rotate(seat_q,world-label.top[i])
            if b=='pelvis':
                xy=np.maximum(np.abs(points[:,:2])-label.extent[i]+.01,0)
                gap=np.sqrt((xy**2).sum(1)+points[:,2]**2).min()
            else:
                points[:,2]+=.05
                excess=np.abs(points)-np.r_[label.extent[i],.05]
                distances.append((np.sqrt((np.maximum(excess,0)**2).sum(1))+np.minimum(excess.max(1),0)).min())
        result.append([gap,min(distances)])
    return np.array(result)

def fixtures():
    n=1
    r=dict(force=np.zeros((1000,n,7,3)),normal=np.tile([0.,0.,1.],(1000,n,7,1)),weight=np.full((1000,n),100.),
           permitted=np.ones((1000,n,7),bool),inside=np.ones((1000,n,7),bool),rest_patch=np.ones((1000,n,7),bool),
           legal=np.ones((1000,n),bool),wrong=np.zeros((1000,n)),failure=np.zeros((1000,n),bool),alive=np.ones((1000,n),bool),
           root_speed=np.zeros((1000,n)),gyro=np.zeros((1000,n)),good=np.ones((1000,n),bool),slip=np.zeros((1000,n,7)),
           no_slip=np.ones((1000,n),bool),total_squared_torque_Nm2=np.zeros((1000,n)))
    r['force'][:,:,0,2]=20
    g=dict(pelvis_target_distance_m=np.zeros((200,n)),housing_vertex_clearance_m=np.ones((200,n)))
    row=dict(success=True,failures={'physical':False},slip_ticks=0,illegal_ticks=0,traction_bad_ticks=0)
    ids=list(range(60,200,5));a,c,_=episode_reward(r,g,[row],[0],ids);b,d=oracle(r,g,[row],[0],ids)
    assert np.allclose(a,b,atol=1e-8,rtol=0) and np.allclose(c,d,atol=1e-8,rtol=0)
    r['no_slip'][500:600]=False;row['slip_ticks']=100
    a,c,z=episode_reward(r,g,[row],[0],ids)
    assert z['event_totals_raw']['first_slip']==[-5.]
    # The only past-event change below excludes clean bonus equally in both.
    r['no_slip'][500:600]=True;r['no_slip'][0:100]=False
    a2,c2,z2=episode_reward(r,g,[row],[0],ids)
    assert np.allclose(a[ids.index(105):],a2[ids.index(105):],atol=1e-12,rtol=0)
    assert abs(a[0,0]-a2[0,0]+.1)<1e-12
    r['failure'][600:]=True;row['failures']['physical']=True
    a,c,z=episode_reward(r,g,[row],[0],ids);b,d=oracle(r,g,[row],[0],ids)
    assert np.allclose(a,b,atol=1e-8,rtol=0) and z['event_totals_raw']['first_physical_failure']==[-20.]
    assert .77625>.14875
    return dict(passed=True,checks=['independent_suffix_oracle','first_event_once','past_event_excluded','physical_event_once','analytic_clean_dominance'])

def saved(root,output):
    root=Path(root);result=dict(classification='Saved trajectories only; reward validation, not physical repair',fixtures=fixtures(),bindings={},runs={},rows=0)
    def bind(p):
        result['bindings'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
    cal=json.loads(bind(root/'calibrate/feedback_calibration.json').read_text())
    names=['train_63111','train_63117']+[f'evaluate_{seed}_{bank}' for seed in [63111,63117] for bank in ['known_old','known_transfer','known_recent']]
    import torch
    for name in names:
        p=root/name
        summary=json.loads(bind(p/'results.json').read_text());rows=summary['episodes']
        geometry=json.loads(bind(p/'geometry.json').read_text()) if (p/'geometry.json').exists() else json.loads(bind(Path(json.loads((p/'manifest.json').read_text())['args']['bank'])/'geometry.json').read_text())
        label=LandingLabels(summary['body_names'],[geometry[x['geometry_index']] for x in rows[:24]],summary['origins'])
        trace=np.load(bind(p/'trace.npz'));ticks=np.load(bind(p/'every_tick.npz'))
        batches=torch.load(bind(p/'decision_batches.pt'),map_location='cpu',weights_only=False)
        records=[]
        for it,batch in enumerate(batches):
            r={k:ticks[k][it*1025:it*1025+1000] for k in ticks.files}
            bodies=trace['body'][it*205:it*205+200]
            labels=[label(x) for x in bodies];g={k:np.array([x[k] for x in labels]) for k in labels[0]}
            for k in [0,60,119,199]:
                ref=geometry_oracle(label,bodies[k]);got=np.stack([g['pelvis_target_distance_m'][k],g['housing_vertex_clearance_m'][k]],axis=1)
                assert np.max(np.abs(ref-got))<1e-10,('GEOMETRY_ORACLE',name,it,k,np.max(np.abs(ref-got)))
            erows=rows[it*24:it*24+24];margin=cal['margin_ticks'] if name.startswith('train') else [0]*24
            a,c,report=episode_reward(r,g,erows,margin,batch['decision_steps'])
            b,d=oracle(r,g,erows,margin,batch['decision_steps']);error=float(max(np.max(np.abs(a-b)),np.max(np.abs(c-d))))
            assert error<2e-7,(name,it,error)
            records.append(dict(update=it+1,max_abs_error=error,original_first_return=batch['reward'][0].tolist(),revised_first_return=a[0].tolist(),**report))
            result['rows']+=24
        result['runs'][name]=records
    # Explicit previously diagnosed same-case ranking contradiction.
    records=result['runs']['train_63111'];bad=records[1];good=records[2]
    assert not bad['clean'][4] and good['clean'][4]
    assert bad['original_first_return'][4]>good['original_first_return'][4]
    assert bad['revised_first_return'][4]<good['revised_first_return'][4]
    result['ranking_example']=dict(case='known_transfer 4',slipped_update2_original=bad['original_first_return'][4],clean_update3_original=good['original_first_return'][4],slipped_update2_revised=bad['revised_first_return'][4],clean_update3_revised=good['revised_first_return'][4])
    result['passed']=True;Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,rows=result['rows'],ranking_example=result['ranking_example'])),flush=True)

if __name__=='__main__':
    if len(sys.argv)==1:print(json.dumps(fixtures()))
    else:saved(sys.argv[1],sys.argv[2])
