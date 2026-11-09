"""CPU boundary checks for the current-sample quiet gate, not physical success."""
import sys,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
sys.path[1:1]=[str(ROOT/'experiments/furniture_repair_20261106_v1'),str(ROOT/'scripts'),str(ROOT/'sitting_task/src')]
from humanoid_sitting.shared_frame import BodyModel,level_rotation
from seat_plan import SeatPlan
from normal_landing import NormalLanding
read=lambda p:json.loads(Path(p).read_text())


def main():
    model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
    inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
    shapes=read(ROOT/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
    reports=[]
    for job in read(HERE/'calibration_jobs.json'):
        name=job['source_case'];folder=Path(job['model_directory'])
        with np.load(ROOT/'outputs/furniture_examples_20261105_v2'/name/'frame_0.npz') as z,np.load(ROOT/'outputs/furniture_examples_20261105_v2'/name/'startup.npz') as start:
            level=level_rotation(z['gravity'].astype(float).copy());shift=level.T@np.array([job['closer_m'],0.,0.])
            geometry=z['geometry'].copy();geometry[5]-=geometry[2:5]@shift;geometry[6:9]+=shift
            plan=SeatPlan(model,inertia,shapes,z['q'],z['gravity'],geometry,z['selected_points']+shift,start['command'][0])
        construction=read(folder/(name+'.plan.json'));path=read(folder/(name+'.complete.json'))
        teacher=NormalLanding(plan,construction,path);tick=len(teacher.approach)
        end=construction['nodes'][-1];q=np.array(end['q']);rt=np.array(end['root']);previous=np.array(end['command'])
        gravity=rt[:3,:3].T@np.array([0.,0.,-1.]);geometry=np.zeros(13)
        geometry[2:5]=rt[:3,:3].T@plan.n;geometry[5]=plan.d+plan.n@rt[:3,3];geometry[11]=1.
        for i in range(24):
            target,signal=teacher.command(tick+i,q,np.zeros(37),gravity,np.zeros(3),geometry,previous)
            assert teacher.travel==0 and teacher.phase=='hover',signal
        target,signal=teacher.command(tick+24,q,np.zeros(37),gravity,np.zeros(3),geometry,previous)
        assert teacher.phase=='landing' and abs(teacher.travel-.00012)<1e-12,signal
        travel=teacher.travel
        qd=np.zeros(37);qd[0]=.051
        teacher.command(tick+25,q,qd,gravity,np.zeros(3),geometry,previous)
        assert teacher.quiet_count==0 and teacher.travel==travel
        stale=geometry.copy();stale[12]=.22
        teacher.command(tick+26,q,np.zeros(37),gravity,np.zeros(3),stale,previous)
        assert teacher.quiet_count==0 and teacher.travel==travel
        wrong=geometry.copy();wrong[5]+=.1
        teacher.command(tick+27,q,np.zeros(37),gravity,np.zeros(3),wrong,previous)
        assert teacher.quiet_count==0 and teacher.travel==travel
        delta=np.clip(.1*(target-previous[:23]),-.003,.003)
        assert abs(delta).max()<=.003
        reports.append(dict(case=job['name'],passed=True,quiet_current_samples=25,
                            moving_stale_and_wrong_plane_hold=True,maximum_normal_step_m=.00012))
    result=dict(passed=bool(reports) and all(r['passed'] for r in reports),cases=reports,
                boundary_fixtures_only=True,physics_steps=0,PPO_updates=0,force_truth_actor_input=False)
    (HERE/'landing_software_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
