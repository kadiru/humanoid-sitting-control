"""One bounded offline convex allocation per frozen pose; never runs physics.

Replace maximum joint-margin allocation with minimum squared corner loads.
Retain every original pose, assumed load, command/effort bound and initial
command; add a5mm individual-foot CoP screen. No gain/rate/arm search.
"""
from pathlib import Path
import copy,datetime,hashlib,json
import numpy as np
from scipy.optimize import linprog,minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'balanced_force_paths_144'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False);source=D/'fixed_foot_probe_134/candidate/fixed_plans_132.json';plans=read(source)
    mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';m=BodyModel(mp);ine=read(ip);mass=np.array(ine['mass']);W=mass.sum()*9.81
    joints={j['name']:j for j in m.joints};limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'};children={}
    for j in m.joints:
        if j['body0']:children.setdefault(j['body0'][0],[]).append(j['body1'][0])
    def descendants(body):
        return {body}.union(*(descendants(c) for c in children.get(body,[])))
    sub={n:descendants(joints[n]['body1'][0]) for n in m.names};summaries=[]
    revision=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,poses=364,solves_per_pose=1,max_slsqp_iterations=200,
        unchanged='All364 poses and geometric records, frozen cooked model/inertia, current-input fixtures, fixed feet/arms/fingers, assumed zero-gap seat-loading schedule, initial forces/commands, actual.03 desired margin, .9999 internal effort target, .1filter and.15rad/s cap. No runtime controller installed.',
        change='Strictly convex minimum squared nonnegative corner loads instead of maximizing minimum joint-command margin. Additional individual-foot5mm CoP reserve. .03001 internal command reserve protects the unchanged actual.03 bound. One allocation per pose, no parameter sweep.',
        physics_ticks=0,optimizer_updates=0,holdout_used=False,source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),source,mp,ip]})
    (D/'balanced_force_revision_144.json').write_text(json.dumps(revision,indent=2))
    for original in plans:
        case=original['case'];plan=copy.deepcopy(original['plan']);ids=plan['moving_ids'];q0=np.array(original['inputs']['q']);cmd0=np.array(original['inputs']['command']);tau0=np.array(plan['initial_static_torque_Nm']);kp=np.array([200,150,150,200,160,160]*2+[40]*4);eff=np.array([300,300,300,300,20,20]*2+[300]*4);lo=np.array([limits[m.names[j]][0] for j in ids]);hi=np.array([limits[m.names[j]][1] for j in ids]);checks=[]
        for k,e in enumerate(plan['path']):
            old=e['load_checks'][0];q=np.array(e['q']);root=np.array(e['root']);T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);fk=m.fk(q);P=np.array(old['corner_world_points']);ss=np.array(old['corner_sides']);oldw=np.array(old['corner_fractions']);C=np.array(e['com']);contact=np.array(old['seat_contact_point']);vertical=old['vertical_seat_fraction'];A=np.stack([np.ones(len(P)),P[:,0],P[:,1]]);b=np.r_[1-vertical,C[:2]-vertical*contact[:2]]
            grav=[];B=[]
            for j in ids:
                n=m.names[j];joint=joints[n];J=T@fk[joint['body0'][0]]@m.frames[n][0];axis=J[:3,'XYZ'.index(joint['axis'])]
                moment=sum((np.cross(apply(T@fk[body],cc)-J[:3,3],[0,0,-mm*9.81]) for body,mm,cc in zip(ine['body_names'],mass,ine['local_com']) if body in sub[n]),np.zeros(3));grav.append(-axis@moment)
                B.append([-axis@np.cross(p-J[:3,3],[0,0,W]) if side+'_ankle_roll_link' in sub[n] else 0. for p,side in zip(P,ss)])
            grav=np.array(grav);B=np.array(B);base=cmd0[ids]+q[ids]-q0[ids]+(grav-tau0)/kp;M=B/kp[:,None]
            U=[M,-M,B/eff[:,None],-B/eff[:,None]];v=[hi-.03001-base,base-lo-.03001,.9999-grav/eff,.9999+grav/eff];footrows={}
            for side in ['left','right']:
                take=ss==side;hull=ConvexHull(P[take,:2]).equations;F=np.zeros((len(hull),len(P)));F[:,take]=hull[:,:2]@P[take,:2].T+hull[:,2,None]+.005;U.append(F);v.append(np.zeros(len(hull)));footrows[side]=hull
            U=np.vstack(U);v=np.concatenate(v)
            feasible=linprog(np.zeros(len(P)),A_ub=U,b_ub=v,A_eq=A,b_eq=b,bounds=(0,1),method='highs',options=dict(primal_feasibility_tolerance=1e-10,dual_feasibility_tolerance=1e-10))
            result=None
            if feasible.success:
                result=minimize(lambda w:.5*np.dot(w,w),feasible.x,jac=lambda w:w,method='SLSQP',bounds=[(0,1)]*len(P),constraints=[dict(type='eq',fun=lambda w:A@w-b,jac=lambda w:A),dict(type='ineq',fun=lambda w:v-U@w,jac=lambda w:-U)],options=dict(ftol=1e-12,maxiter=200))
            w=(result.x if result is not None else oldw).copy()
            if k==0:w=oldw.copy()
            tau=grav+B@w;cmd=cmd0.copy();cmd[ids]=base+M@w
            if k==0:tau=tau0.copy();cmd=cmd0.copy()
            margin=float(min(np.r_[cmd[ids]-lo,hi-cmd[ids]]));eu=float(max(abs(tau)/eff));eq=float(max(abs(A@w-b)));violation=float(max(0.,max(U@w-v)));cp={}
            for side,hull in footrows.items():
                take=ss==side;f=w[take].sum();pos=(w[take,None]*P[take,:2]).sum(0)/f;cp[side]=dict(vertical_bw=float(f),position=pos.tolist(),margin_m=float(min(-(hull[:,:2]@pos+hull[:,2]))))
            passed=bool(feasible.success and (k==0 or result.success) and eq<1e-8 and min(w)>=-1e-10 and margin>=.03 and eu<=1 and min(z['margin_m'] for z in cp.values())>=.005-1e-10)
            old.update(corner_fractions=w.tolist(),static_torque_Nm=tau.tolist(),command=cmd.tolist(),min_command_margin_rad=margin,maximum_effort_utilization=eu,max_normalized_equilibrium_residual=eq,passed=passed,
                force_objective='minimum squared corner loads with individual-foot5mm CoP reserve',solver_success=bool(feasible.success and (k==0 or result.success)),solver_message=str(result.message if result is not None else feasible.message),individual_foot_cop=cp,max_constraint_violation=violation)
            e.update(min_command_margin_rad=margin,maximum_effort_utilization=eu,all_endpoint_model_checks_passed=bool(e['model_pose_passed'] and passed));checks.append(passed)
        cmd=np.array([e['load_checks'][0]['command'] for e in plan['path']]);tau=np.array([e['load_checks'][0]['static_torque_Nm'] for e in plan['path']]);summary=dict(case=case,geometry_unchanged=True,geometry_passed=plan['summary']['geometry_passed'],commands_passed=all(checks),failed_knots=[i for i,b in enumerate(checks) if not b],minimum_command_margin_rad=min(e['min_command_margin_rad'] for e in plan['path']),maximum_effort_utilization=max(e['maximum_effort_utilization'] for e in plan['path']),first_command_change_max_rad=float(abs(cmd[1]-cmd[0]).max()),max_adjacent_ankle_torque_change_Nm=float(abs(np.diff(tau[:,[4,10]],axis=0)).max()),first_ankle_pitch_torque_Nm=tau[:2,[4,10]].tolist(),command_polyline_time_s=float(np.max(abs(np.diff(cmd,axis=0)),axis=1).sum()/.15))
        plan['summary'].update(summary);plan['summary']['time_gate_passed']=summary['command_polyline_time_s']<=13.5;plan['summary']['endpoint_model_passed']=plan['path'][-1]['all_endpoint_model_checks_passed'];plan['model_passed']=all(checks) and plan['summary']['geometry_passed'];plan['summary']['actual_filter_checked']=False
        (O/f'plan_{case}.json').write_text(json.dumps(dict(case=case,plan=plan,inputs=original['inputs'],source_sha256=revision['source_sha256']),indent=2));summaries.append(summary);print(json.dumps(summary),flush=True)
    (O/'results.json').write_text(json.dumps(dict(revision=revision,paths=summaries,physics_ticks=0,optimizer_updates=0,physical_admission=False),indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
