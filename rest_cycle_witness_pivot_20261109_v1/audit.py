"""Independent gate recomputation; no solver success is an admission."""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply
from audit_mesh_clearance import audit as mesh_audit

def audit(plan,result,shapes,furniture,allowed,frame):
    m=plan.mechanics;named=[];worst={};balance=[];gaps=[];patch=[];tilt=[];foot=[];bias=[]
    nodes=result['nodes']+result['return_nodes'][1:]
    for i,node in enumerate(nodes):
        q=np.asarray(node['q']);rt=np.asarray(node['root']);s=m.state(plan.x(q,rt),node['load'])
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
    checks=dict(feet=max(foot)<=2e-5,balance=min(balance)>=-1e-8,plane_nonpenetration=min(gaps)>=-1e-6,
        patch_reserve=min(patch)>=-1e-8,tilt=min(tilt)>=-1e-8,base_wrench=max(bias)<=1e-8,
        first20s_hold_reserve=result['nominal_rest_settle_s']<14.4,return20s=result['nominal_return_settle_s']<=20,
        partial_support=result['rest_load']>=.05,
        original_command_limits=all(c['filter_error']<1e-10 and c['increment']<=.003+1e-12 and c['desired_margin']>=.03 and c['fingers_fixed'] for c in [a,b]))
    basis=frame['root_rotation']@plan.R0.T
    clearance=mesh_audit(plan.model,shapes,furniture,allowed,nodes,basis,frame['root_position'])
    checks['full_mesh_clearance']=clearance['passed']
    work=m.independent_work_audit(plan.x(np.asarray(nodes[len(result['nodes'])-1]['q']),np.asarray(nodes[len(result['nodes'])-1]['root'])),result['rest_load'])
    checks['independent_virtual_work']=work['passed']
    return dict(model_admitted=all(checks.values()),tests={k:bool(v) for k,v in checks.items()},
      minimum_balance_margin=min(balance),minimum_plane_gap_m=min(gaps),minimum_patch_margin_m=min(patch),maximum_foot_error=max(foot),
      forward=a,backward=b,clearance=clearance,virtual_work=work,nodes=named,physical_success=False,model_only=True)
