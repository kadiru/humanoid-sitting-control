"""Input checks preceding any mechanics or mesh admission.

Checks are independent of the path constructor. No optimizer or simulator is
imported, no missing values are imputed, and no trajectory is repaired here.
"""
import numpy as np


def check_plan(result, lower, upper, initial_command):
    errors = []
    def reject(reason):
        errors.append(reason)

    def array(value, shape, label):
        try:
            a = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            reject(label+':not_numeric'); return None
        if a.shape != shape or not np.isfinite(a).all():
            reject(label+':shape_or_nonfinite'); return None
        return a

    lo = array(lower, (23,), 'lower')
    hi = array(upper, (23,), 'upper')
    initial = array(initial_command, (37,), 'initial_command')
    if errors:
        return dict(passed=False, errors=errors)
    if np.any(lo+.06 > hi):
        reject('empty_original_command_box')
    paths = {}
    for direction, node_key in (('forward', 'nodes'), ('backward', 'return_nodes')):
        nodes = result.get(node_key)
        schedule = result.get(direction)
        if not isinstance(nodes, list) or not 2 <= len(nodes) <= 1001 or not isinstance(schedule, dict):
            reject(direction+':missing_or_unbounded_path'); continue
        parsed = []
        for i, node in enumerate(nodes):
            label = f'{node_key}[{i}]'
            if not isinstance(node, dict):
                reject(label+':not_a_node'); continue
            try:
                q = array(node['q'], (37,), label+'.q')
                command = array(node['command'], (37,), label+'.command')
                root = array(node['root'], (4,4), label+'.root')
                scalars = array([node[k] for k in ('load','gap','marker_gap','model_effort_ratio')],
                                (4,), label+'.scalars')
                array(node['model_cop'], (3,), label+'.cop')
                array(node['model_tau'], (23,), label+'.torque')
            except KeyError as error:
                reject(label+':missing_'+str(error)); continue
            if any(a is None for a in (q, command, root, scalars)):
                continue
            if not (0. <= scalars[0] <= .8):
                reject(label+':nonphysical_support_fraction')
            if scalars[3] < 0 or node.get('phase') not in ('approach','landing'):
                reject(label+':invalid_effort_or_phase')
            # Original physical joint margin. Producer bounds are not evidence
            # that the independent reader received a valid configuration.
            if np.any(q[:23] <= lo+.005) or np.any(q[:23] >= hi-.005):
                reject(label+':original_joint_margin')
            r = root[:3,:3]
            if (not np.allclose(root[3], [0,0,0,1], rtol=0, atol=1e-10) or
                not np.allclose(r.T@r, np.eye(3), rtol=0, atol=1e-8) or
                abs(np.linalg.det(r)-1) > 1e-8):
                reject(label+':not_a_rigid_transform')
            if not np.array_equal(command[23:], initial[23:]):
                reject(label+':finger_command_changed')
            parsed.append((q, command, root))
        if len(parsed) != len(nodes):
            continue
        try:
            count = len(schedule['commands'])
            if not 1 <= count <= 1000:
                reject(direction+':command_count'); continue
            commands = array(schedule['commands'], (count,37), direction+'.commands')
            desired = array(schedule['desired'], (count,23), direction+'.desired')
            seconds = array(schedule['seconds'], (), direction+'.seconds')
        except (KeyError, TypeError) as error:
            reject(direction+':missing_schedule_'+str(error)); continue
        if any(a is None for a in (commands, desired, seconds)):
            continue
        if abs(float(seconds)-count*.02) > 1e-10:
            reject(direction+':time_does_not_match_commands')
        nominal_key = 'nominal_rest_settle_s' if direction == 'forward' else 'nominal_return_settle_s'
        nominal = array(result.get(nominal_key), (), nominal_key)
        if nominal is not None and abs(float(nominal)-(count*.02+(1.5 if direction == 'forward' else 2.5))) > 1e-10:
            reject(direction+':nominal_time_does_not_match_commands')
        previous = np.vstack([parsed[0][1], commands[:-1]])
        predicted = previous[:,:23]+np.clip(.1*(desired-previous[:,:23]), -.003, .003)
        if (np.max(abs(predicted-commands[:,:23])) > 1e-10 or
            np.max(abs(commands-previous)) > .003+1e-12 or
            np.any(desired < lo+.03) or np.any(desired > hi-.03) or
            not np.array_equal(commands[:,23:], np.broadcast_to(initial[23:], (count,14)))):
            reject(direction+':original_action_contract')
        cursor = 0
        for i, (_, command, _) in enumerate(parsed[1:], 1):
            matches = np.flatnonzero(np.max(abs(commands[cursor:]-command), axis=1) < 1e-12)
            if not len(matches):
                reject(f'{direction}:node_{i}_not_in_ordered_commands'); break
            count_segment = int(matches[0])+1
            start = parsed[i-1][1]
            fractions = np.arange(1,count_segment+1,dtype=float)/count_segment
            expected = start[None,:]+fractions[:,None]*(command-start)[None,:]
            if np.max(abs(commands[cursor:cursor+count_segment]-expected)) > 1e-12:
                reject(f'{direction}:segment_{i-1}_does_not_match_declared_interpolation')
            cursor += count_segment
        if cursor != count:
            reject(direction+':commands_after_final_node')
        # The executor reserves 25 quiet samples at each of four forward
        # boundaries, a final five-second hold, then release and standing quiet.
        reserve = 4*25+250 if direction == 'forward' else 25+100
        if count+reserve > 1000:
            reject(direction+':original_window_has_no_qualification_reserve')
        paths[direction] = parsed
    if len(paths) == 2:
        f, b = paths['forward'], paths['backward']
        for x, y, label in ((f[-1], b[0], 'rest_join'), (b[-1], f[0], 'standing_join')):
            if any(not np.array_equal(a, c) for a, c in zip(x,y)):
                reject(label+':pose_or_command_discontinuity')
        if not np.array_equal(f[0][1], initial):
            reject('initial_command_binding')
    for key in ('nominal_rest_settle_s','nominal_return_settle_s','rest_load','rest_model_effort_ratio'):
        if key not in result:
            reject('missing_'+key)
        else:
            array(result[key], (), key)
    if not errors and result['rest_load'] != result['nodes'][-1]['load']:
        reject('rest_load_binding')
    return dict(passed=not errors, errors=errors, physical_success=False,
                scope='Finite plan inputs, ordered command association and original limits only')
