"""Timestamped evaluation evidence, preserving the upstream contact verdict.

This module is prohibited in actor/teacher inputs. Raw per-contact buffers are
stored by tuple index without inventing a resultant friction force per point.
The seven aggregate channels and every threshold remain upstream and unchanged.
"""
import numpy as np
from contact_adapter import MeshMetrics, array


def select_contact_arrays(values, capacity):
    counts = array(values[4]).reshape(-1)
    starts = array(values[5]).reshape(-1)
    if len(counts) != 1 or len(starts) != 1:
        raise ValueError('EXPECTED_ONE_BODY_ONE_FILTER')
    count, start = int(counts[0]), int(starts[0])
    if count >= capacity or count < 0 or (count and start < 0):
        raise ValueError('CONTACT_BUFFER_SATURATED_OR_INVALID')
    selected = []
    for value in values[:4]:
        data = array(value)
        if count and start + count > len(data):
            raise ValueError('CONTACT_BUFFER_RANGE')
        chunk = (data[start:start+count] if count else data[:0]).copy()
        if not np.isfinite(chunk).all():
            raise ValueError('NONFINITE_CONTACT_BUFFER')
        selected.append(chunk.tolist())
    return dict(count=count, start=start, tuple_components_0_to_3=selected)


class RecordedMeshMetrics(MeshMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evidence = []
        self.recorded_steps = set()

    def __call__(self, raw):
        result = super().__call__(raw)
        step = int(raw.sim.get_physics_step_count())
        dt = float(raw.sim.get_physics_dt())
        self.frames[-1].update(physics_step=step, physics_clock_s=step*dt,
                               metrics_call_index=len(self.frames)-1)
        if step not in self.recorded_steps:
            row = dict(physics_step=step, physics_clock_s=step*dt, physics_dt_s=dt,
                       mapping_frame_index=len(self.frames)-1, channels=[])
            for sensor_name, body_name, rest in self.support.CONTACTS:
                sensor=raw.scene[sensor_name]
                buffers=select_contact_arrays(sensor.contact_view.get_contact_data(dt=dt),
                                              sensor.cfg.max_contact_data_count_per_prim)
                row['channels'].append(dict(sensor=sensor_name, body=body_name,
                                             rest_patch=bool(rest), **buffers))
            row['aggregate_force_w_N']=array(result[0]['force_w']).tolist()
            row['scoring_normal_w']=array(result[0]['normal_w']).tolist()
            row['scoring_inside_patch']=array(result[0]['inside_patch']).tolist()
            row['scoring_failure']=array(result[0]['failure']).tolist()
            self.evidence.append(row);self.recorded_steps.add(step)
        return result
