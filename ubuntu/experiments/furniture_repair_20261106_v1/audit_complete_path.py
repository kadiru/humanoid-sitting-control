"""Independent discrete filter/rate and normal-contact geometry checks."""
import numpy as np
from humanoid_sitting.shared_frame import apply

def audit(plan, construction, complete):
    previous=np.array(construction['nodes'][0]['command']);peak=0.;margins=[];mismatch=0.
    limits={j['name']:np.deg2rad(j['limits']) for j in plan.model.joints if j['name'] in plan.model.names}
    for desired,target in zip(complete['approach_desired'],complete['approach_commands']):
        desired=np.array(desired);target=np.array(target)
        applied=previous+np.clip(.1*(desired-previous),-.003,.003)
        peak=max(peak,float(abs(applied-previous).max()));mismatch=max(mismatch,float(abs(applied-target).max()))
        for i,name in enumerate(plan.model.names[:23]):margins.append(min(desired[i]-limits[name][0],limits[name][1]-desired[i]))
        previous=applied
    tangents=[];foot_errors=[];normals=[];commands=[]
    for node in complete['landing']:
        q=np.array(node['q']);rt=np.array(node['root']);fk=plan.model.fk(q)
        pts=apply(rt,plan.mesh['pelvis']);gap=pts@plan.n+plan.d;tip=pts[gap.argmin()]
        tangents.append([tip@plan.e,tip@plan.b]);normals.append(float(gap.min()))
        commands.append(node['command'])
        for side,anchor in zip(('left','right'),plan.feet):foot_errors.append(float(np.linalg.norm((rt@fk[side+'_ankle_roll_link'])[:3,3]-anchor[:3,3])))
    tangential=float(np.max(np.linalg.norm(np.array(tangents)-tangents[0],axis=1)))
    landing_rates=np.max(np.abs(np.diff(commands,axis=0)),axis=1)/(.001/.006)
    checks=dict(discrete_filter_exact=mismatch<1e-10,original_rate_limit=peak<=.003+1e-12,
                desired_command_margin=min(margins)>=.03,normal_approach=tangential<.002,
                feet_fixed=max(foot_errors)<.003,landing_command_rate=float(max(landing_rates))<=.15,
                original_hold_time_reserved=complete['nominal_latest_settled_s']<14.4)
    return dict(passed=all(checks.values()),tests=checks,maximum_increment_rad=peak,
                minimum_desired_margin_rad=min(margins),normal_path_tangent_drift_m=tangential,
                nominal_latest_settled_s=complete['nominal_latest_settled_s'],
                final_model_gap_m=normals[-1],actual_quiet_gate_may_delay=True,
                missing_before_physics=['full collision clearance','live sensor controller and reset checks'],
                simulation_admitted=False,physical_success=False)
