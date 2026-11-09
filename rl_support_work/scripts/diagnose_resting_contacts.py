"""Replay unchanged resting policies and log the exact first-failure sources."""
import hashlib
import json
from pathlib import Path
import shutil
import torch
import run_resting_posture as base

original_metrics = base.metrics
original_env = base.RestEnv
active_env = None


def instrumented_metrics(raw):
    m, detail, flags = original_metrics(raw)
    env = active_env
    if env is None:
        return m, detail, flags
    due = m['failure'] & env.alive & (env.warming == 0)
    if not due.any():
        return m, detail, flags
    r = raw.scene['robot']; contact = raw.scene['all_contacts']
    history = contact.data.net_normal_forces_w_history.torch
    peaks = history.norm(dim=-1).max(1).values
    allowed = {body for _, body, _ in base.CONTACTS}
    limits = r.data.joint_pos_limits.torch[:, env.ids]
    q = r.data.joint_pos.torch[:, env.ids]
    margin = torch.minimum(q-limits[..., 0], limits[..., 1]-q)
    for i in due.nonzero().flatten().tolist():
        joints = []
        for k in torch.argsort(margin[i])[:5].tolist():
            j = env.ids[k]
            joints.append(dict(name=r.joint_names[j], margin_rad=float(margin[i,k]),
                q=float(q[i,k]), lower=float(limits[i,k,0]), upper=float(limits[i,k,1]),
                command=float(env.previous[i,j]), reset_command=float(env.bank['command'][i,j])))
        bodies = []
        for k in torch.argsort(peaks[i],descending=True).tolist():
            if float(peaks[i,k]) < 1.: continue
            name=contact.body_names[k]; j=r.body_names.index(name)
            bodies.append(dict(name=name, allowed_patch_body=name in allowed,
                peak_normal_N=float(peaks[i,k]), min_vertical_normal_N=float(history[i,:,k,2].min()),
                position_local=(r.data.body_pos_w.torch[i,j]-raw.scene.env_origins[i]).tolist()))
        filtered = {}
        for side in ('left','right'):
            for part in ('foot','knee'):
                s=raw.scene[f'{side}_{part}_seat'].data
                filtered[f'{side}_{part}_seat']=float(s.normal_force_matrix_w_history.torch[i].sum((1,2)).norm(dim=-1).max())
        other = {}
        for sensor, body, rest in base.CONTACTS:
            if rest:
                s=raw.scene[sensor].data
                other[body]=float((s.net_normal_forces_w.torch[i].sum(0)-s.normal_force_matrix_w.torch[i].sum((0,1))).norm())
        row=dict(case=i,attempt=int(env.attempt[i]),t_s=(env.t+1)*base.DT,
            height_m=raw.cfg.geometry_bank[int(base.furniture.indices(raw)[i])]['height'],
            flags={k:bool(v[i]) for k,v in flags.items()},wrong_force_N=float(detail['wrong_force_N'][i]),
            prior_unintended_N=float(base.full_transition.unintended_peak_force(raw)[i]),
            body_contacts=bodies, filtered_seat_contacts_N=filtered,rest_body_nonseat_normal_N=other,
            nearest_joint_limits=joints, body_names=r.body_names)
        with (env.out/'first_failures.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
    return m, detail, flags


class DiagnosticEnv(original_env):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        source=Path(__file__).resolve()
        shutil.copy2(source,self.out/'source'/source.name)
        base.write_json(self.out/'diagnostic_source.json',dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),runtime_unchanged=True))

    def step(self, actions):
        global active_env
        active_env=self
        try:
            return super().step(actions)
        finally:
            active_env=None


base.metrics=instrumented_metrics
base.RestEnv=DiagnosticEnv
if __name__=='__main__':
    base.main()
