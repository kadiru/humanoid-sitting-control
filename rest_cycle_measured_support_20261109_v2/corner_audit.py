"""Independent gate recomputation; no solver success is an admission."""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply
from audit_mesh_clearance import audit as mesh_audit
from plan_contract import check_plan
from interpolation_audit import audit_interpolation

def audit(plan,result,shapes,furniture,allowed,frame,*,clearance_evaluator=None):
    m=plan.mechanics;named=[];worst={};balance=[];gaps=[];patch=[];tilt=[];foot=[];bias=[]
    contract=check_plan(result,m.lower,m.upper,plan.previous)
    if not contract['passed']:
        return dict(model_admitted=False,tests=dict(input_contract=False),input_contract=contract,
                    physical_success=False,model_only=True)
    # The declared geometric path must satisfy the same foot/plane gates
    # between knots. Endpoint admission alone cannot certify its interpolation.
    interpolation={}
    try:
        for direction,key in (('forward','nodes'),('backward','return_nodes')):
            interpolation[direction]=audit_interpolation(plan.model,result[key],result[direction],
                m.foot_bodies,m.feet,m.mesh['pelvis'],plan.n,plan.d)
    except (ValueError,TypeError,KeyError) as error:
        return dict(model_admitted=False,tests=dict(input_contract=True,interpolation=False),
                    interpolation_error=str(error),physical_success=False,model_only=True)
    if not all(value['passed'] for value in interpolation.values()):
        return dict(model_admitted=False,tests=dict(input_contract=True,interpolation=False),
                    interpolation=interpolation,physical_success=False,model_only=True)
    nodes=result['nodes']+result['return_nodes'][1:]
    rest_effort_ratio=None
    for i,node in enumerate(nodes):
        q=np.asarray(node['q']);rt=np.asarray(node['root']);s=m.state(plan.x(q,rt),node['load'])
        # Python min/max may silently ignore a NaN occurring after a finite
        # element. Reject every computed state before aggregating any margins.
        expected_state_shapes=dict(foot_error=(12,),com=(3,),tau=(23,),cop=(3,),
                                   foot_forces=(2,3),foot_points=(2,3),base_residual=(6,))
        if any(np.asarray(s[k]).shape!=shape or not np.isfinite(np.asarray(s[k],float)).all()
               for k,shape in expected_state_shapes.items()):
            return dict(model_admitted=False,tests=dict(input_contract=True,finite_mechanics=False),
                        invalid_mechanics_node=i,physical_success=False,model_only=True)
        if i==len(result['nodes'])-1:
            if not np.isfinite(m.standing_effort) or m.standing_effort<=0:
                return dict(model_admitted=False,tests=dict(finite_standing_effort=False),
                            physical_success=False,model_only=True)
            rest_effort_ratio=float(np.asarray(s['tau'])@np.asarray(s['tau'])/m.standing_effort)
        foot.append(float(abs(s['foot_error']).max()));tilt.append(float(rt[2,2]-np.cos(np.deg2rad(38))))
        p=apply(rt,m.mesh['pelvis']);heights=p@plan.n+plan.d;gaps.append(float(heights.min()))
        # Actual lowest2mm patch gets the unchanged20mm reserve; no soft penalty.
        low=p[heights<=heights.min()+.002];uv=np.c_[low@m.e,low@m.b]
        if node['phase']=='landing':patch.append(float((-(uv@m.patch[:,:2].T+m.patch[:,2])-.02).min()))
        for eq,point,F in zip(m.foot_polygons,s['foot_points'],s['foot_forces']):
            balance.extend([float((-(eq[:,:2]@point[:2]+eq[:,2])-.01).min()),float(F[2]/m.W-.05),float((.5*F[2]-np.linalg.norm(F[:2]))/m.W)])
        named.append(dict(node=i,phase=node['phase'],gap_m=gaps[-1],load=node['load'],base_wrench_residual=float(abs(s['base_residual']).max())))
        bias.append(float(abs(s['base_residual']).max()))
    def commands(nodes,scheduled):
        cmds=np.asarray(scheduled['commands']);desired=np.asarray(scheduled['desired']);previous=np.vstack([nodes[0]['command'],cmds[:-1]])
        pred=previous[:,:23]+np.clip(.1*(desired-previous[:,:23]),-.003,.003)
        return dict(filter_error=float(abs(pred-cmds[:,:23]).max()),increment=float(abs(cmds-previous).max()),
          desired_margin=float(np.minimum(desired-m.lower,m.upper-desired).min()),fingers_fixed=bool(np.array_equal(cmds[:,23:],np.broadcast_to(plan.previous[23:],cmds[:,23:].shape))))
    a=commands(result['nodes'],result['forward']);b=commands(result['return_nodes'],result['backward'])
    checks=dict(input_contract=True,interpolation=True,finite_mechanics=True,feet=max(foot)<=2e-5,balance=min(balance)>=-1e-8,plane_nonpenetration=min(gaps)>=-1e-6,
        patch_reserve=min(patch)>=-1e-8,tilt=min(tilt)>=-1e-8,base_wrench=max(bias)<=1e-8,
        first20s_hold_reserve=result['nominal_rest_settle_s']<14.4,return20s=result['nominal_return_settle_s']<=20,
        partial_support=result['rest_load']>=.05,
        effort_hypothesis=rest_effort_ratio<1.,
        original_command_limits=all(c['filter_error']<1e-10 and c['increment']<=.003+1e-12 and c['desired_margin']>=.03 and c['fingers_fixed'] for c in [a,b]))
    from shapely.geometry import Polygon
    from scipy.spatial import ConvexHull
    corner_margins=[];coverage=[]
    for node in nodes:
        if node['phase']!='landing':continue
        p=apply(np.asarray(node['root']),m.mesh['pelvis']);height=p@plan.n+plan.d;cut=height.min()+.002
        cap=list(p[height<=cut]);aa,bb=plan.edges.T;cross=(height[aa]-cut)*(height[bb]-cut)<0
        aa,bb=aa[cross],bb[cross];u=(cut-height[aa])/(height[bb]-height[aa])
        cap.extend(p[aa]+u[:,None]*(p[bb]-p[aa]));cap=np.asarray(cap)
        uv=np.c_[cap@m.e,cap@m.b];hull=Polygon(uv[ConvexHull(uv).vertices])
        coverage.append(plan.observed.covers(hull));corner_margins.append(float(hull.distance(plan.observed.boundary)-.02))
    checks['entire_observed_corner_polygon']=all(coverage) and min(corner_margins)>=-1e-8
    basis=frame['root_rotation']@plan.R0.T
    clearance=(mesh_audit if clearance_evaluator is None else clearance_evaluator)(
        plan.model,shapes,furniture,allowed,nodes,basis,frame['root_position'])
    checks['full_mesh_clearance']=clearance['passed']
    work=m.independent_work_audit(plan.x(np.asarray(nodes[len(result['nodes'])-1]['q']),np.asarray(nodes[len(result['nodes'])-1]['root'])),result['rest_load'])
    checks['independent_virtual_work']=bool(work['passed'] and
        np.isfinite([work['maximum_joint_torque_error_Nm'],work['maximum_base_wrench_residual']]).all() and
        work['maximum_joint_torque_error_Nm']<=2e-6 and work['maximum_base_wrench_residual']<=1e-8)
    return dict(model_admitted=all(checks.values()),tests={k:bool(v) for k,v in checks.items()},
      recomputed_rest_model_effort_ratio=rest_effort_ratio,
      minimum_corner_margin_m=min(corner_margins),minimum_balance_margin=min(balance),minimum_plane_gap_m=min(gaps),minimum_patch_margin_m=min(patch),maximum_foot_error=max(foot),
      forward=a,backward=b,interpolation=interpolation,clearance=clearance,virtual_work=work,nodes=named,physical_success=False,model_only=True)
