"""Independent Torch CPU replay of108; no physics or optimizer."""
from pathlib import Path
import datetime, hashlib, json, argparse
import numpy as np
import torch
from humanoid_sitting.shared_frame import BodyModel

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'contact_filtered_schedule_audit_109'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=int,choices=[0,6],required=True);args=parser.parse_args();O=D/f'staged_fixed_foot_filter_audit_v2_{args.case}_123';O.mkdir(exist_ok=False)
    src=D/f'staged_fixed_foot_filtered_{args.case}_122/results.json';data=read(src)
    for p,h in data['source_sha256'].items():assert sha(p)==h,p
    npz=D/f'staged_fixed_foot_filtered_{args.case}_122/command_schedule.npz';a=np.load(npz)
    m=BodyModel(D/'cooked_model_50/model.json');names=m.names
    ids=[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]
    ids += [names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']]
    assert np.array_equal(ids,a['moving_ids'])
    limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'}
    lo=torch.tensor([limits[names[j]][0]+.03 for j in ids],dtype=torch.float32)
    hi=torch.tensor([limits[names[j]][1]-.03 for j in ids],dtype=torch.float32)
    assert np.array_equal(lo.numpy(),a['lo']) and np.array_equal(hi.numpy(),a['hi'])
    Q=np.array([e['load_checks'][0]['command'] for e in read(D/'staged_fixed_foot_contact_120'/f'case_{args.case}'/'results.json')['path']])
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(Q,axis=0),ord=np.inf,axis=1))]
    pos=a['source_arc'];assert pos[0]==0 and pos[-1]==arc[-1] and np.all(np.diff(pos)>0)
    # Independent vectorized piecewise interpolation of the saved path coordinates.
    reconstructed=np.stack([np.interp(pos,arc,Q[:,j]) for j in range(len(names))],axis=1).astype(np.float32)
    npinterp_exact=bool(np.array_equal(reconstructed,a['desired']));npinterp_max=float(abs(reconstructed-a['desired']).max());npinterp_count=int(np.count_nonzero(reconstructed!=a['desired']))
    # Vectorized reproduction of the specified a+fraction*(b-a) operation order.
    # np.interp uses different floating arithmetic; retain its exact-gate result.
    k=np.clip(np.searchsorted(arc,pos,side='right')-1,0,len(Q)-2)
    fraction=np.divide(pos-arc[k],arc[k+1]-arc[k],out=np.zeros_like(pos),where=arc[k+1]!=arc[k])
    ordered=(Q[k]+fraction[:,None]*(Q[k+1]-Q[k])).astype(np.float32)
    assert np.array_equal(ordered,a['desired'])
    fixed=[j for j in range(len(names)) if j not in ids]
    fingers=[j for j,n in enumerate(names) if any('_'+v+'_joint' in n for v in ['zero','one','two','three','four','five','six'])]
    assert len(fingers)==14
    previous=torch.from_numpy(a['applied'][0].copy());clipped=0
    for k,raw0 in enumerate(a['raw_target'],1):
        raw=torch.from_numpy(raw0.copy())
        assert torch.isfinite(raw).all()
        if bool(((raw[ids]<lo)|(raw[ids]>hi)).any()):clipped+=1
        assert torch.equal(raw[fixed],previous[fixed])
        previous[ids]=previous[ids]+(.1*(torch.minimum(torch.maximum(raw[ids],lo),hi)-previous[ids])).clamp(min=-.003,max=.003)
        assert torch.equal(previous,torch.from_numpy(a['applied'][k]))
    assert clipped==0 and np.array_equal(a['applied'][-1],Q[-1].astype(np.float32))
    assert np.array_equal(a['applied'][:,fingers],np.broadcast_to(Q[0,fingers].astype(np.float32),a['applied'][:,fingers].shape))
    assert np.array_equal(a['t'],np.arange(len(a['applied']))*.02)
    runtime=Path('scripts/run_seat_load_ppo.py');code=runtime.read_text()
    assert "clamp(min=limits[...,0]+.03,max=limits[...,1]-.03)" in code
    assert "(.1*(desired-self.previous[:,self.ids])).clamp(min=-cap,max=cap)" in code
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,
             scope='Independent vectorized source-segment reconstruction using specified a+fraction*(b-a) arithmetic is exactly equal afterfloat32 conversion. Historical np.interp exact comparison is separately retained; no tolerance changed or saved command altered. Torch CPU replay of every inherited clamp/filter/increment operation is exactly equal to saved122 applied commands. Fixed14fingers and endpoint exact. No GPU/simulation/physical feasibility or minimum-time proof.',
             case=args.case,command_ticks=len(a['raw_target']),duration_s=float(a['t'][-1]),inverse_target_clipped_ticks=clipped,
             desired_polyline_float32_exact=True,npinterp_float32_exact=npinterp_exact,npinterp_max_difference_rad=npinterp_max,npinterp_different_entries=npinterp_count,applied_torch_cpu_exact=True,endpoint_float32_exact=True,
             max_float32_applied_increment_rad=float(abs(np.diff(a['applied'],axis=0)).max()),
             actual_filter_arithmetic_passed=True,time_gate=bool(a['t'][-1]<=13.5),
             physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False,
             source_sha256={str(p.resolve()):sha(p) for p in [src,npz,runtime,Path(__file__)]})
    (O/'independent_audit.json').write_text(json.dumps(out,indent=2))
    (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()
