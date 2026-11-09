"""Independent motor-axis measurement admission after one recorded lean."""
import os,sys,json,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'sitting_task/src')]
import numpy as np
from humanoid_sitting.shared_frame import BodyModel,level_rotation
from foot_wrench_observer import FootWrenchObserver
read=lambda p:json.loads(Path(p).read_text())

def main():
    out=ROOT/'outputs'/HERE.name/'motor_axis_readout';d=dict(np.load(out/'trajectory.npz'))
    model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
    inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
    observer=FootWrenchObserver(model,inertia);pred=[];condition=[];normal=[]
    assert len(d['q_pre'])==1000
    for t in range(1000):
        q=d['q_pre'][t,0];gravity=d['gravity_pre'][t,0];tau=d['motor_axis_effort_pre'][t,0]
        foot,cond=observer.estimate(q,tau,gravity)
        pred.append([-foot[:,0].sum(),0.,observer.weight-foot[:,1].sum()]);condition.append(cond)
        plane=d['obs'][t,0,322:325];n=level_rotation(gravity.astype(float))@plane;normal.append(n/np.linalg.norm(n))
    pred=np.asarray(pred);condition=np.array(condition);normal=np.array(normal)[1:]
    corrected=pred[1:]-pred[0]
    start=np.load(out/'frame_0.npz');basis=start['root_rotation']@level_rotation(start['gravity'].astype(float)).T
    actual=d['physical_force_w'][:-1,0,:5].sum(1)@basis
    up=np.array([0.,0.,1.])-normal*normal[:,2,None];up/=np.linalg.norm(up,axis=1)[:,None]
    estimate=np.c_[(corrected*normal).sum(1),(corrected*up).sum(1)]
    target=np.c_[(actual*normal).sum(1),(actual*up).sum(1)]
    error=np.mean(abs(estimate[-250:]-target[-250:]),axis=0)
    pass_measurement=bool(condition.max()<200 and error.max()<.02*observer.weight)
    result=dict(passed=pass_measurement,condition_max=float(condition.max()),final250_mean_abs_error_N=error.tolist(),
                final_predicted_normal_shear_N=estimate[-1].tolist(),final_actual_normal_shear_N=target[-1].tolist(),
                startup_sensor_bias_N=pred[0].tolist(),sensor_only_prediction=True,validation_threshold_BW=.02,
                simulator_steps_added_by_audit=0,ppo_updates=0,
                scope='Measurement validation only; unchanged physical rollout success is audited separately.')
    with (out/'motor_axis_audit.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
    if not pass_measurement:sys.exit(3)
if __name__=='__main__':main()
