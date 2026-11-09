"""Independent recomputation of model admission, not physical success."""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply


def audit(plan, result):
    tests = {}; feet = []; coms = []; insets = []; gaps = []; margins = []
    limits = {j['name']: np.deg2rad(j['limits']) for j in plan.model.joints if 'limits' in j}
    for node in result['nodes']:
        q = np.asarray(node['q']); rt = np.asarray(node['root']); fk = plan.model.fk(q)
        for side, anchor in zip(('left', 'right'), plan.feet):
            foot = rt @ fk[side+'_ankle_roll_link']
            feet.append(max(np.linalg.norm(foot[:3, 3]-anchor[:3, 3]),
                            np.linalg.norm(Rotation.from_matrix(anchor[:3, :3].T @ foot[:3, :3]).as_rotvec())))
        c = np.average([apply(rt@fk[b], p) for b,p in zip(plan.inertia['body_names'],plan.inertia['local_com'])],
                       axis=0,weights=plan.inertia['mass'])
        coms.append(c[0])
        p = apply(rt,plan.mesh['pelvis']); depth=p@plan.n+plan.d; gaps.append(float(depth.min()))
        patch=p[depth<depth.min()+.002]
        if node is result['nodes'][-1]:
            uv=np.c_[patch@plan.e,patch@plan.b]
            insets.extend((-(uv@plan.hull.equations[:,:2].T+plan.hull.equations[:,2])).ravel())
        for i,name in enumerate(plan.model.names[:23]):
            margins.append(min(node['command'][i]-limits[name][0],limits[name][1]-node['command'][i]))
    tests['bilateral_feet_fixed']=max(feet)<.003
    tests['balance_in_foot_support']=min(coms)>=plan.foot_bounds[0]+.01 and max(coms)<=plan.foot_bounds[1]-.01
    tests['hover_clearance']=min(gaps)>=.012-.002
    tests['contact_patch_observed_interior']=bool(min(insets)>.02)
    tests['command_margin_unchanged']=min(margins)>=.03
    tests['initial_command_exact']=bool(np.array_equal(result['nodes'][0]['command'],plan.previous))
    tests['time_for_original_hold']=result['earliest_settled_seconds']<14.4
    return dict(passed=all(tests.values()),tests=tests,max_foot_error=max(feet),
                min_command_margin=min(margins),min_patch_inset=float(min(insets)),
                com_range=[min(coms),max(coms)],earliest_settled_seconds=result['earliest_settled_seconds'],
                missing_before_physics=['full observed-obstacle clearance','final landing/load controller','live quiet-state gate'],
                simulation_admitted=False,physical_success=False)
