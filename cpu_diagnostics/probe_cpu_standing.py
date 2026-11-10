"""One fixed CPU posture sensor capture, with no trajectory or tuning.

Runs only after the native model/reset audit passes. Saves every physics-step
reading, including failures. A standing diagnostic cannot certify a rest cycle.
"""
import argparse
import json
import resource
import time
from pathlib import Path
import numpy as np
from isolated_runtime import start
from import_cpu_g1 import run


def probe(stage, simulation, view, robot, names, indices, contract, out):
    from pxr import PhysicsSchemaTools
    assert contract['physics_dt'] == .005
    motion = None
    if 'measurement_path' in contract:
        # A frozen path is only a sensor-validation stimulus. Its timing never
        # counts as measured contact, release, or completion of a task.
        motion = np.load(contract['measurement_path'])
        assert contract['physics_steps'] == 4*len(motion['desired']) <= 8000
        assert motion['desired'].shape[1] == 23
    else:
        assert contract['physics_steps'] == 1000
    links = list(robot.shared_metatype.link_names)
    assert links[0] == 'pelvis', 'UNEXPECTED_ROOT_LINK_ORDER'
    feet = [links.index(side+'_ankle_roll_link') for side in ('left', 'right')]
    paths = ['/Robot/'+name for name in links]
    filters = ['/RestFloor'] + (['/Support'] if 'support' in contract else [])
    contacts = view.create_rigid_contact_view(paths, [filters for _ in paths], 4096)
    assert contacts.sensor_count == 44
    arrays = {key: [] for key in ('q','qd','tau','pose','velocity','normal','friction','sleeping','command')}
    initial_q = robot.get_dof_positions()[0, indices].copy()
    initial_root = robot.get_root_transforms().copy()
    targets = robot.get_dof_position_targets().copy()
    velocity_targets = robot.get_dof_velocity_targets().copy()
    effort_targets = robot.get_dof_actuation_forces().copy()
    ids = np.array([0], dtype=np.uint32)
    body_path = PhysicsSchemaTools.sdfPathToInt('/Robot/pelvis')
    previous = targets[0, indices].copy()
    if motion is not None:
        assert np.array_equal(previous, motion['initial'])
        lo, hi = motion['lower'], motion['upper']
        assert np.isfinite(motion['desired']).all()
        assert (motion['desired'] >= lo+.03).all() and (motion['desired'] <= hi-.03).all()
    progress = dict(physics_steps=0, complete_cycles=0, sensor_repair=False,
                    GPU_devices_exposed=False)
    (out/'progress.json').write_text(json.dumps(progress))
    try:
        for tick in range(contract['physics_steps']):
            if motion is not None and tick % 4 == 0:
                desired = motion['desired'][tick//4]
                previous[:23] += np.clip(np.float32(.1)*(desired-previous[:23]),
                                        np.float32(-.003),np.float32(.003))
                assert np.max(abs(previous-motion['command'][tick//4])) <= 2e-7
                targets[0, indices] = previous
            # IsaacLab writes the unchanged motor targets before every physics
            # substep. Omitting those writes permits sleep and changes sensor
            # publication; matching values alone is not runtime parity.
            robot.set_dof_actuation_forces(effort_targets, ids)
            robot.set_dof_position_targets(targets, ids)
            robot.set_dof_velocity_targets(velocity_targets, ids)
            simulation.simulate(.005, tick*.005)
            simulation.fetch_results()
            progress['physics_steps'] = tick+1
            arrays['q'].append(robot.get_dof_positions()[0, indices].copy())
            arrays['qd'].append(robot.get_dof_velocities()[0, indices].copy())
            arrays['tau'].append(robot.get_dof_projected_joint_forces()[0, indices].copy())
            arrays['pose'].append(robot.get_link_transforms()[0].copy())
            arrays['velocity'].append(robot.get_link_velocities()[0].copy())
            arrays['normal'].append(contacts.get_contact_force_matrix(.005).copy())
            f, points, counts, offsets = contacts.get_friction_data(.005)
            assert int(counts.sum()) < len(f), 'FRICTION_BUFFER_POSSIBLY_SATURATED'
            force = np.zeros((44,len(filters),3), dtype=float)
            for i in range(44):
                for j in range(len(filters)):
                    a, n = int(offsets[i,j]), int(counts[i,j])
                    assert n >= 0 and (n == 0 or 0 <= a <= len(f)-n)
                    force[i,j] = f[a:a+n].sum(0)
            arrays['friction'].append(force)
            arrays['sleeping'].append(simulation.is_sleeping(
                simulation.get_attached_stage(), body_path))
            arrays['command'].append(previous.copy())
            if tick%100 == 99:
                (out/'progress.json').write_text(json.dumps(progress))
            # Stop on a severe failure; preserve the entire prefix.
            pose = arrays['pose'][-1]
            if not np.isfinite(pose).all() or pose[0,2] < .28:
                progress['abort'] = 'NONFINITE_OR_FALL'
                break
            if not all(np.isfinite(arrays[k][-1]).all() for k in
                       ('q','qd','tau','velocity','normal','friction')):
                progress['abort'] = 'NONFINITE_NATIVE_SENSOR'
                break
            if motion is not None:
                normal_force = arrays['normal'][-1]
                floor_wrong = [i for i in range(44) if i not in feet]
                permitted_support = [links.index(x) for x in contract['permitted_support_bodies']]
                support_wrong = [i for i in range(44) if i not in permitted_support]
                if (np.linalg.norm(normal_force[floor_wrong,0],axis=-1).max() > 10 or
                    np.linalg.norm(normal_force[support_wrong,1],axis=-1).max() > 10):
                    progress['abort'] = 'ILLEGAL_CONTACT_OVER_ORIGINAL_10N_LIMIT'
                    break
    finally:
        data = {key: np.asarray(value) for key,value in arrays.items()}
        np.savez_compressed(out/'trajectory.npz', **data, initial_q=initial_q,
            initial_root=initial_root, joint_names=names, link_names=links,
            mass=robot.get_masses().copy(), com=robot.get_coms().copy(),filters=filters)
        progress['recorded_samples'] = {key:len(value) for key,value in arrays.items()}
        (out/'progress.json').write_text(json.dumps(progress, indent=2))
    ankles = [i for i,name in enumerate(names) if 'ankle' in name]
    last = slice(-200,None)
    weight = float(robot.get_masses().sum())*9.81
    dq = np.diff(np.vstack([initial_q,data['q']]),axis=0)/.005
    wrong = [i for i in range(44) if i not in feet]
    summary = dict(progress,
        native_ankle_peak_last_second=float(abs(data['qd'][last][:,ankles]).max()),
        encoder_difference_peak_last_second=float(abs(dq[last][:,ankles]).max()),
        encoder_difference_is_diagnostic_only=True,
        native_quiet_controls_last_second=int(np.sum(np.max(abs(data['qd'][last][3::4,:23]),axis=1)<=.05)),
        foot_load_mean_last_second_bw=float(data['normal'][last][:,feet,0,2].sum(1).mean()/weight),
        sleeping_samples=int(data['sleeping'].sum()),
        nonfoot_floor_peak_N=float(np.linalg.norm(data['normal'][:,wrong,0],axis=-1).max()),
        final_root_height_m=float(data['pose'][-1,0,2]),
        scope=('Frozen unloading sensor stimulus' if motion is not None else 'Single accepted posture reset')+
              '; no camera, loaded/release admission, or complete cycle')
    if 'support' in contract:
        normal = np.asarray(contract['support']['normal'])
        summary['support_normal_mean_last_second_bw'] = float(
            ((data['normal'][last][:,:,1]+data['friction'][last][:,:,1]).sum(1)@normal).mean()/weight)
    (out/'standing_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('contract',type=Path)
    parser.add_argument('output',type=Path)
    a=parser.parse_args();a.output.mkdir(exist_ok=False)
    started=time.monotonic();app=start()
    try:
        run(a.contract,a.output,probe=probe)
    except BaseException as exc:
        (a.output/'failure.json').write_text(json.dumps({'error':repr(exc)}))
        raise
    finally:
        import omni.physx
        omni.physx.get_physx_simulation_interface().detach_stage()
        r=resource.getrusage(resource.RUSAGE_SELF)
        (a.output/'resources.json').write_text(json.dumps(dict(
            wall_s=time.monotonic()-started,process_CPU_s=r.ru_utime+r.ru_stime,
            peak_RSS_KiB=r.ru_maxrss,GPU_devices_exposed=False)))
        app.shutdown()


if __name__=='__main__':
    main()
