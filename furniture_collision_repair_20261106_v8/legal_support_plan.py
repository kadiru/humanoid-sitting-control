import time
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply
from sagittal_static_bias import static_bias
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'furniture_collision_repair_20261106_v6/legal_support_plan.py'
spec=importlib.util.spec_from_file_location('support_v6',path);v6=importlib.util.module_from_spec(spec);spec.loader.exec_module(v6)
class SeatPlan(v6.SeatPlan):
    def solve(self):
        self.deadline = time.monotonic() + 45.
        q, root, fk, pelvis, h, tip, com = self.state(self.z0)
        # The endpoint residual directly fits the actual legal support footprint.
        # Its22mm observed inset and the independent20mm audit remain mandatory.
        # A pelvis-only footprint at a thigh-selected lane is not this problem.
        target=float(tip @ self.e)
        eq=self.hull.equations;lane=float(tip @ self.b);forward=eq[:,0]>1e-5
        edge=float(np.min((-eq[forward,2]-eq[forward,1]*lane)/eq[forward,0]))
        balance = self.foot_bounds[0]+.02
        # A fresh endpoint solve supplies an arm reference for the path. It
        # cannot import a previous failed command vector or relax constraints.
        self.arm_reach = 1.
        endpoint = self.fit(self.z0.copy(), .012, target, balance)
        self.arm_goal = endpoint[9:11].copy()
        seed = self.z0.copy(); nodes = []
        initial_gap, initial_tangent, initial_balance = float(h.min()), float(tip @ self.e), float(com[0])
        # Solve fresh measured-geometry constraints; no prior failed action vector.
        for s in np.linspace(0, 1, 33):
            self.arm_reach = s
            self.progress = s
            gap = (1-s)*initial_gap + s*.012
            tangent = (1-s)*initial_tangent + s*target
            cop = (1-s)*initial_balance + s*balance
            if s:
                seed = self.fit(seed, gap, tangent, cop)
            q, root, fk, pelvis, heights, tip, com = self.state(seed)
            floor_root = root.copy(); floor_root[2, 3] -= self.floor
            pose = np.r_[floor_root[:3, 3], Rotation.from_matrix(root[:3, :3]).as_quat()]
            bias, static = static_bias(self.model, self.inertia, q, pose, float(tip[0]), 0.)
            arm_bias=[]
            for joint in self.armids:
                potentials=[]
                for sign in (-1,1):
                    trial=q.copy();trial[joint]+=sign*1e-5
                    tf=self.model.fk(trial)
                    potentials.append(sum(m*9.81*apply(root@tf[b],c)[2] for b,m,c in
                        zip(self.inertia['body_names'],self.inertia['mass'],self.inertia['local_com'])))
                arm_bias.append((potentials[1]-potentials[0])/(2e-5*40.))
            arm_bias=np.asarray(arm_bias)
            if not nodes:
                arm_preload=self.previous[self.armids]-q[self.armids]-arm_bias
                preload = self.previous[self.legids] - q[self.legids] - bias
            cmd = self.previous.copy(); cmd[self.legids] = q[self.legids] + bias + (1-s)*preload
            cmd[self.armids]=q[self.armids]+arm_bias+(1-s)*arm_preload
            nodes.append(dict(q=q.tolist(), root=root.tolist(), command=cmd.tolist(),
                              phase='approach', gap=float(heights.min()),
                              contact_tangent=float(tip @ self.e), balance=float(com[0]), static=static))
        nodes[0]['command'] = self.previous.tolist()
        # Conservative command arc timing with a half-second speed ramp at each end.
        commands = np.array([x['command'] for x in nodes])
        arc = np.r_[0., np.cumsum(np.max(abs(np.diff(commands, axis=0)), axis=1))]
        approach_seconds = float(arc[-1]/.125 + 1.)
        return dict(nodes=nodes, command_arc=arc.tolist(), approach_seconds=approach_seconds,
                    earliest_contact_seconds=approach_seconds+.5+.012/.006,
                    earliest_settled_seconds=approach_seconds+.5+.012/.006+1.,
                    target_tangent=target, target_balance=balance, target_rule='joint_endpoint_actual_legal_support_22mm_observed_inset', arm_endpoint=self.arm_goal.tolist(),
                    sensor_front_edge=float(edge), measured_floor=self.floor,
                    foot_bounds=self.foot_bounds, surface_normal=self.n.tolist(),
                    surface_offset=self.d, independent_audit_required=True,
                    physical_success=False)
