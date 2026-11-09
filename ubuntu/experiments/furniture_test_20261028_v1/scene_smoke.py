"""One bounded scene/standing-reset check, only when the 4090 is idle.

Does not train or score a policy. The existing slab scorer is deliberately not
used to label arbitrary furniture. This is the required first physics check.
"""
import os,sys,json,subprocess,argparse,time
from pathlib import Path
GPU='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent

def assert_idle():
    query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    jobs=[x for x in query.splitlines() if x.split(',')[0].strip()==GPU]
    assert not jobs,'4090_BUSY: no scene launch while another process owns the training GPU'

def main():
    assert_idle()
    os.environ['CUDA_VISIBLE_DEVICES']=GPU
    os.environ['OMP_NUM_THREADS']='1'
    sys.path.insert(0,str(ROOT/'scripts'))
    import numpy as np
    import torch
    import run_resting_posture as base
    import run_supported_postures as support
    from scene_adapter import load_case,make_cfg
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',required=True)
    base.add_launcher_args(p);a=p.parse_args();a.device='cuda:0';a.num_envs=1;a.seed=64201;a.mode='fixed';a.enable_cameras=True;a.depth_check=True
    c=load_case(a.case);out=Path(a.output);out.mkdir(exist_ok=False)
    validation=json.loads((HERE/'static_validation.json').read_text());assert validation['passed']
    import hashlib
    check=next(x for x in validation['checks'] if x['case']==a.case)
    assert hashlib.sha256((HERE/'scenes'/a.case/'furniture.usda').read_bytes()).hexdigest()==check['file_sha256']
    cfg=make_cfg(a,c);bank=np.load(ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting/bank.npz')
    states={k:bank['reset_'+k][:1] for k in ['q','qd','command','root_pose','root_velocity']}
    records=[];started=time.monotonic()
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;raw.reset();support.restore(raw,states)
        robot=raw.scene['robot'];cmd=torch.as_tensor(states['command'],device=raw.device,dtype=torch.float32)
        init_steps=raw.sim.get_physics_step_count();assert init_steps<=1000
        try:
            with torch.no_grad():
                for tick in range(125): # .5s warmup + 2s standing observation, no retries
                    assert time.monotonic()-started<5400
                    raw.step(cmd);assert not raw.reset_buf.any()
                    d=robot.data
                    tilt=torch.acos((-d.projected_gravity_b.torch[:,2]).clamp(-1,1))
                    height=d.root_pos_w.torch[:,2]-raw.scene.env_origins[:,2]
                    records.append(dict(tick=tick+1,root_height=float(height[0]),tilt_deg=float(torch.rad2deg(tilt)[0]),
                        body=d.body_pose_w.torch.cpu().numpy().tolist(),contacts=raw.scene['all_contacts'].data.net_normal_forces_w.torch.cpu().numpy().tolist()))
                    m,detail,flags=support.metrics(raw)
                    # The original predicates are valid for this separated
                    # standing check only if no body is loading furniture.
                    assert not (m['force_w'][:,:5].norm(dim=-1)>.01*m['bodyweight_N'][:,None]).any(),'UNEXPECTED_STANDING_FURNITURE_CONTACT'
                    records[-1]['physical_flags']={k:bool(v[0]) for k,v in flags.items()}
                    assert not m['failure'].any(),'STANDING_PHYSICAL_FAILURE'
            clock=raw.sim.get_physics_step_count();raw.sim.render_context.update_scene_state(clock)
            cam=raw.scene['rear_camera'];base.acquire_current_camera(cam);assert raw.sim.get_physics_step_count()==clock
            depth=cam.data.output['distance_to_image_plane'].torch.cpu().numpy()
            np.savez_compressed(out/'rear_depth.npz',depth=depth,intrinsics=cam.data.intrinsic_matrices.torch.cpu().numpy())
            assert raw.sim.get_physics_step_count()<=init_steps+125*4
            (out/'scene_check.json').write_text(json.dumps(dict(case=c,standing_reset_check_passed=True,
                controls=125,initialization_physics_steps=init_steps,physics_steps=raw.sim.get_physics_step_count(),
                GPU=GPU,policy_trial=False,clean_score=None,contact_scoring_validated=False,records=records),indent=2)+'\n')
        except Exception as e:
            (out/'failure.json').write_text(json.dumps(dict(error=repr(e),records=records,clean_score=None),indent=2)+'\n');raise
        finally:raw.close()
if __name__=='__main__':main()
