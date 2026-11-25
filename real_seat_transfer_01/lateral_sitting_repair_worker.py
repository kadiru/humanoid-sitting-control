"""Keep verified routes and apply the trace-guided planted-foot repairs."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def exact_module(path):
    spec = importlib.util.spec_from_file_location('saved_sitting_skills', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slow_clock(tick, settings):
    start = settings['slow_start']
    span = settings['slow_end']-start
    fade = .24*span
    x = float(np.clip(tick-start,0.,span))
    head = min(x/fade,1.)
    tail = np.clip((x-span+fade)/fade,0.,1.)
    area = (fade*(head**3-.5*head**4)
            + np.clip(x-fade,0.,span-2*fade)
            + fade*(tail-tail**3+.5*tail**4))
    return tick-settings['slow_fraction']*area


def choose_camera_route(asset, pose, selector):
    """Choose among verified signed poses using only the registered standing view."""
    axis = 1 if abs(pose[1]) >= selector['lateral_boundary_m'] else 2
    side = np.sign(pose[axis])
    seen = [r for r in selector['templates'][asset]
            if r['axis'] == axis and np.sign(r['pose'][axis]) == side]
    if not seen:
        return None, None
    distance = np.linalg.norm((np.asarray([r['pose'] for r in seen]) - pose) /
                              selector['scale'], axis=1)
    nearest = int(np.argmin(distance))
    err = float(distance[nearest])
    return (seen[nearest] if err <= selector['max_distance'] else None), err


def repair_routes(config, routes, downstream, selector=None, bind_shared=None):
    from chair_contact_motion import ChairContactMotion
    from rescan_sitting import smooth

    previous = ChairContactMotion.apply
    path = config['planted_descent']
    handoff_settings = {}
    captures = {}

    def apply(motion, learner, obs, tick, nominal):
        blend = {}
        if tick:
            for row, patch in routes.items():
                feedback = motion.rescan_feedback.get(row)
                if patch.get('wait_for_upright') and feedback is not None:
                    # Use the planted controller's tilt-checked hold capture
                    # instead of the earlier load-only rim hold for this route.
                    feedback['settled'] = 0
            for row, settings in handoff_settings.items():
                state = motion.rescan_feedback.get(row)
                if state is None:
                    continue
                if row not in captures and state['hold_target'] is not None:
                    captures[row] = (tick, int(state['clock'])-1, state['hold_target'].copy())
                if row not in captures:
                    continue
                captured, cursor, anchor = captures[row]
                age = tick-captured-settings['pressure_dwell_ticks']
                if age < 0:
                    state['clock'] = cursor
                    continue
                duration = settings['pressure_ramp_ticks']
                u = min(age/duration,1.)
                # Take up the remaining extension at a gradually increasing
                # speed after the supported capture has settled.
                progress = duration*(u**3-.5*u**4) + max(0,age-duration)
                state['clock'] = min(cursor+progress,settings['pressure_final_cursor'])
                state['hold_target'] = None
                state['settled'] = 0
                blend[row] = (float(smooth(u)), anchor)
        out = previous(motion, learner, obs, tick, nominal)
        for row,(mix,anchor) in blend.items():
            out[row] = (1-mix)*out.new_tensor(anchor)+mix*out[row]
        if tick == 0:
            ev = learner.env.out
            measured = json.loads((ev/'camera_routes.json').read_text())
            if selector is not None:
                selected = []
                for row, observed in enumerate(measured):
                    pick, distance = None, None
                    if observed['registration_ok'] and observed['standing']:
                        pick, distance = choose_camera_route(observed['asset'],
                                                            observed['relative_pose'], selector)
                    if pick is not None and pick['repair'] is not None:
                        routes[row] = pick['repair']
                    if pick is not None and pick['shared']:
                        bind_shared(row)
                    selected.append(dict(row=row, anchor=pick['case'] if pick else None,
                                         distance=distance, mode='shared' if pick and pick['shared'] else
                                         'repair' if row in routes else 'staged'))
                (ev/'signed_camera_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
            handoff_settings.update({row:s for row,s in routes.items() if 'pressure_dwell_ticks' in s})
            actual = []
            for row, patch in routes.items():
                observed = measured[row]
                if not (observed['registration_ok'] and observed['standing']):
                    continue
                if row not in path['rows']:
                    path['rows'].append(row)
                key = str(row)
                path['treatments'][key] = {**path['treatments'].get(key,{}), **patch}
                actual.append(dict(row=row, recipe=path['treatments'][key], family=observed['family']))
            (ev/'lateral_routes_applied.json').write_text(json.dumps(actual,indent=2)+'\n')
        if tick == 999 and handoff_settings:
            captured = {str(row):dict(tick=tick0,path_tick=cursor)
                        for row,(tick0,cursor,_) in captures.items()}
            (learner.env.out/'supported_handoff.json').write_text(json.dumps(captured,indent=2)+'\n')
        return out

    ChairContactMotion.apply = apply
    return downstream()


def retime_paths(routes, downstream):
    from chair_contact_motion import ChairContactMotion

    slowed = {}
    previous = ChairContactMotion.apply

    def apply(motion, learner, obs, tick, nominal):
        out = previous(motion, learner, obs, tick, nominal)
        if tick == 0:
            slowed.update({row:s for row,s in routes.items() if 'slow_start' in s})
        for row, settings in slowed.items():
            if row not in motion.planted_paths or tick < settings['slow_start']:
                continue
            if motion.planted_actions[row]['hold'] is not None:
                continue
            k = slow_clock(tick,settings)
            a = int(k)
            path = motion.planted_paths[row]
            v = path[a]+(k-a)*(path[min(a+1,999)]-path[a])
            out[row] = out.new_tensor(v)
        return out

    ChairContactMotion.apply = apply
    return downstream()


def steady_shoes(routes, downstream):
    from chair_contact_motion import ChairContactMotion
    from rescan_sitting import smooth

    targets, lateral = {}, {}
    first = 1000
    previous = ChairContactMotion.apply

    def apply(motion, learner, obs, tick, nominal):
        nonlocal first
        out = previous(motion, learner, obs, tick, nominal)
        env = learner.env
        if tick == 0:
            targets.update({row:s for row,s in routes.items() if s.get('shoe_pitch_start') is not None})
            lateral.update({row:s['shoe_roll_bias'] for row,s in targets.items() if 'shoe_roll_bias' in s})
            first = min((s['shoe_pitch_start'] for s in targets.values()), default=1000)
            motion.shoe_pitch_trim = {row:np.zeros(2) for row in targets}
            motion.shoe_roll_trim = {row:np.zeros(2) for row in lateral}
            motion.shoe_pitch_log = {row:[] for row in targets}
        if tick < first:
            return out
        # The planner imports USD through the calibrated model. Wait for Kit
        # to initialize before loading it, as in the planted descent worker.
        from stance23_planner import chains, legs, forward

        for row, setting in targets.items():
            start = setting['shoe_pitch_start']
            if tick < start:
                continue
            data = env.robot.data
            q = data.joint_pos.torch[row].cpu().numpy()
            up = -data.projected_gravity_b.torch[row].cpu().numpy()
            ramp = smooth((tick-start)/40.)
            pitch_bias = setting.get('shoe_pitch_bias', (0., 0.))
            measured = []
            for side,(chain,ids) in enumerate(zip(chains,legs)):
                pose, jac = forward(chain,q[ids])
                forward_axis = pose[:3,0]
                dot = float(np.clip(-up @ forward_axis,-.999,.999))
                pitch = np.arcsin(dot)
                sensitivity = -up @ np.cross(jac[3:,4],forward_axis)/np.sqrt(1-dot**2)
                error = pitch - np.clip(pitch,-.006,.006)
                rate = -.9*error/sensitivity if abs(sensitivity)>.55 else 0.
                desired = ramp * np.clip(rate + pitch_bias[side], -.030, .030)
                trim = motion.shoe_pitch_trim[row]
                trim[side] += np.clip(desired-trim[side],-.00075,.00075)
                out[row,ids[4]] += float(trim[side])
                measured.append(float(pitch))
                if row in lateral and lateral[row][side]:
                    # Counter the measured lateral edge loading on this shoe.
                    axis = pose[:3, 1]
                    roll = np.arcsin(np.clip(up @ axis, -.999, .999))
                    correction = ramp * np.clip(lateral[row][side] - .9*roll, -.022, .022)
                    roll_trim = motion.shoe_roll_trim[row]
                    roll_trim[side] += np.clip(correction-roll_trim[side],-.00075,.00075)
                    out[row,ids[5]] += float(roll_trim[side])
                    measured.extend([float(roll),float(roll_trim[side])])
            motion.shoe_pitch_log[row].append([tick,*measured,*motion.shoe_pitch_trim[row]])
        if tick==999:
            np.savez_compressed(env.out/'shoe_pitch_feedback.npz',
                                **{str(row):v for row,v in motion.shoe_pitch_log.items()})
        return out

    ChairContactMotion.apply = apply
    return downstream()


if __name__=='__main__':
    original, flag, profile, launcher = sys.argv[1:5]
    assert flag=='--arm-worker'
    sys.path[:0]=[str(Path(original).parent),str(Path(launcher).parent),str(Path(launcher).parent/'trainer')]
    output=Path(sys.argv[sys.argv.index('--output')+1])
    config=json.loads((output.parent/'step_config.json').read_text())
    skills=json.loads((output.parent/'skills.json').read_text())
    repair=json.loads((output.parent/'repair.json').read_text())
    selector=repair.get('camera_selector')
    routes={int(row):settings for row,settings in repair['routes'].items()}
    shared_rows=set(repair['use_shared'])

    from controlled_seat_step_worker import install_step
    from fresh_sitting_skills_worker import attach_camera_routes
    from pelvis_marker_worker import install
    from planted_descent import attach as planted
    from rescan_sitting import attach as rescan
    from sitting_load_effort import attach as rest
    from sitting_skills import ALIGN, DESCEND, TRANSFER, REST, Decisions, attach_wire_template
    import evaluate_armchair_pair as pair
    import seat_alignment

    shared = exact_module(repair['shared']['implementation']).Decisions(repair['shared'])

    class RepairDecisions(Decisions):
        def __init__(self, settings):
            # The shared model uses all four stages. Dispatch only its protected
            # route to that model; the staged incumbent still starts at transfer.
            super().__init__({**settings,'phases':[ALIGN,DESCEND,TRANSFER,REST]})
            self.guarded=True
            self.approved={}
            self.blocked=[]

        def choose(self,row,phase,values,enabled=True):
            if row in shared_rows:
                action = shared.choose(row,phase,values,enabled)
                self.records[row,phase] = shared.records[row,phase]
                return action
            if row in routes or phase in (ALIGN,DESCEND):
                return super().choose(row,phase,values,False)
            allowed = self.approved.get(row,{})
            if phase==REST and str(TRANSFER) in allowed:
                prior = self.records.get((row,TRANSFER))
                enabled = enabled and prior is not None and prior[1]==allowed[str(TRANSFER)]
            action = super().choose(row,phase,values,enabled)
            if action and action!=allowed.get(str(phase),0):
                features,_,_,used=self.records[row,phase]
                self.records[row,phase]=features,0,0.,used
                self.blocked.append(dict(row=int(row),phase=int(phase),proposed=int(action)))
                return 0
            return action

        def save(self,file):
            super().save(file)
            (file.parent/'suppressed_options.json').write_text(json.dumps(self.blocked,indent=2)+'\n')

    decisions=RepairDecisions(skills)
    correction=seat_alignment.correction
    heading=[]

    def bind_shared(row):
        # Alignment already ran with the feet planted. Transfer the same
        # measured alignment features into the shared actor's state.
        shared.choose(row,ALIGN,decisions.records[row,ALIGN][0][4:7],False)
        shared_rows.add(row)

    def align(prediction):
        row=len(heading)
        delta=correction(prediction)
        heading.append(prediction)
        action=decisions.choose(row,ALIGN,delta/seat_alignment.UNITS,abs(delta[0])>=.012)
        if selector is not None or routes.get(row,{}).get('standing'):
            return np.zeros_like(delta)
        return delta*(1.,.8,0.,1.15)[action]

    seat_alignment.correction=align
    install_step(output,config)
    before=pair.trace_contacts

    def run(launch,options):
        def after(l,o):
            def finish():
                return retime_paths(routes,lambda:rest(config,lambda:
                    steady_shoes(routes,lambda:decisions.attach(lambda:before(l,o)))))

            def route():
                return attach_camera_routes(config,decisions,
                    lambda: repair_routes(config,routes,lambda:planted(config,finish),selector,bind_shared))

            return attach_wire_template(config,route)

        return rescan(config,launch,options,after)

    pair.trace_contacts=run
    install(Path(original),Path(profile),Path(launcher),sys.argv[5:])
