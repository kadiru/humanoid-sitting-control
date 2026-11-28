"""Execute admitted clearance paths while preserving the full existing controller stack."""
import json
from pathlib import Path
import sys

import numpy as np

from sitting_clearance_descent import ANCHOR_LOAD, END, FEET, KP, START, smooth


def attach(downstream, config):
    from chair_contact_motion import ChairContactMotion

    previous = ChairContactMotion.apply
    paths = {}
    with np.load(config['paths'],allow_pickle=False) as archive:
        for row in config['rows']:
            prefix = f'{row}_'
            paths[row] = {key.removeprefix(prefix):archive[key] for key in archive.files if key.startswith(prefix)}

    def apply(motion, learner, obs, tick, nominal):
        out = previous(motion,learner,obs,tick,nominal)
        if tick<START:
            return out
        env = learner.env
        data = env.robot.data
        if tick==START:
            # The mass and mesh dependencies need the initialized USD runtime.
            from rearstep_kin import MassModel

            motion.corridor_mass = MassModel()
            motion.corridor_state = {}
            motion.corridor_log = np.full((1000,len(out),7),np.nan,dtype=np.float32)
            body = data.body_pose_w.torch.cpu().numpy()
            feet = [env.robot.body_names.index(name) for name in FEET]
            origins = env.raw.scene.env_origins.cpu().numpy()
            q = data.joint_pos.torch.cpu().numpy()
            gravity = data.projected_gravity_b.torch.cpu().numpy()
            evidence = []
            for row,path in paths.items():
                foot_error = float(np.linalg.norm(body[row,feet,:3]-origins[row]-path['feet'][:,:3],axis=1).max())
                joint_error = float(np.max(np.abs(q[row,:23]-path['start_q'][:23])))
                admitted = bool(env.alive[row]) and foot_error<.004 and joint_error<.012
                evidence.append(dict(row=row,admitted=admitted,foot_error_m=foot_error,joint_error_rad=joint_error))
                if admitted:
                    forward = (path['basis'][:,0]@projection(body[row,0,3:])).copy()
                    up = -gravity[row]
                    forward -= (forward@up)*up
                    forward /= np.linalg.norm(forward)
                    bias = env.previous[row,:23].cpu().numpy()-path['q'][0]-path['ff'][0]
                    motion.corridor_state[row] = dict(cursor=0.,eligible=tick,age=0,old=None,
                                                     velocity=0.,bias=bias,trim=0.,heading=forward)
            (env.out/'clearance_capture.json').write_text(json.dumps(evidence,indent=2)+'\n')
        q = data.joint_pos.torch.cpu().numpy()
        gravity = data.projected_gravity_b.torch.cpu().numpy()
        seat_sensor = env.raw.scene['pelvis_seat'].data
        loads = seat_sensor.normal_force_matrix_w.torch.sum((1,2))[:,2].cpu().numpy()
        bw = data.body_mass.torch.sum(1).cpu().numpy()*9.81
        for row,state in motion.corridor_state.items():
            if not bool(env.alive[row]):
                continue
            path = paths[row]
            cursor = state['cursor']
            u = min(cursor/(END-START),1.)
            param = u*(len(path['q'])-1)
            j = min(int(param),len(path['q'])-2)
            a = param-j

            def interp(key):
                return (1.-a)*path[key][j]+a*path[key][j+1]

            up = -gravity[row]
            x = state['heading']
            x = x-up*(x@up)
            x /= np.linalg.norm(x)
            frame = np.stack((x,np.cross(up,x),up))
            cm,feet = motion.corridor_mass.estimate(q[row],frame)
            measured = float(cm[0]-feet[:,0].mean())
            target = float(interp('com')@path['basis'][:,0])
            old = measured if state['old'] is None else state['old']
            speed = .82*state['velocity']+.18*(measured-old)*50.
            state.update(old=measured,velocity=speed)
            load = max(0.,loads[row]/bw[row])
            phase = smooth((tick-START)/55.)
            demand = phase*float(np.clip(1.35*(measured-target)+.33*max(0.,speed-.035),0.,.024))*(1.-smooth(load/.10))
            state['trim'] += float(np.clip(demand-state['trim'],-.001,.001))
            if u>=1.:
                state['age'] += 1
            preloaded = smooth(state['age']/90.)
            use_load = float(np.clip(load/ANCHOR_LOAD,0.,1.))*smooth((u-.91)/.09)
            cmd = interp('q')+interp('ff')+(1.-phase)*state['bias']+use_load*path['release']+preloaded*path['press']
            # The same original target filter, rate limit and physical guards apply
            # after this command. This trim only uses remaining pitch effort.
            cap = np.maximum(0.,18.5/KP[15:17]-interp('ff')[15:17])
            cmd[15:17] += np.minimum(cap,state['trim'])
            out[row,:23] = out.new_tensor(cmd)
            rate = .6 if measured-target+.20*max(speed,0.)>.030 and load<.07 else 1.
            state['cursor'] = min(END-START,cursor+rate)
            motion.corridor_log[tick,row] = [cursor,measured,target,speed,state['trim'],load,preloaded]
        if tick==999:
            np.savez_compressed(env.out/'clearance_feedback.npz',data=motion.corridor_log,
                columns=('progress','com_relative_m','target_relative_m','forward_speed_m_s',
                         'ankle_trim_rad','seat_fraction','press_fraction'))
        return out

    ChairContactMotion.apply = apply
    return downstream()


def projection(quaternion):
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(quaternion).as_matrix()


def main():
    original, flag, _, launcher = sys.argv[1:5]
    assert flag=='--arm-worker'
    sys.path[:0] = [str(Path(original).parent),str(Path(launcher).parent),str(Path(launcher).parent/'trainer')]
    output = Path(sys.argv[sys.argv.index('--output')+1])
    import off_axis_sitting_pose_worker
    import sitting_skills

    config = json.loads((output.parent/'clearance_route.json').read_text())

    class ClearanceDecisions(sitting_skills.Decisions):
        def attach(self, downstream):
            return super().attach(lambda: attach(downstream,config))

    sitting_skills.Decisions = ClearanceDecisions
    off_axis_sitting_pose_worker.main()


if __name__=='__main__':
    main()
