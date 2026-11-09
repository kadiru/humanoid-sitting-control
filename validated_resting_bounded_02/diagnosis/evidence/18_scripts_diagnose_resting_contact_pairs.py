"""Attribute the hip-housing contacts to seat versus floor, without control changes."""
import json
import diagnose_resting_contacts as diag
base=diag.base
original_cfg=base.make_cfg
original_metrics=base.metrics

def make_cfg(args,geometry):
    cfg=original_cfg(args,geometry)
    for side in ('left','right'):
        setattr(cfg.scene,f'diagnostic_{side}_hip',base.ContactSensorCfg(
            prim_path=f'{{ENV_REGEX_NS}}/Robot/{side}_hip_pitch_link',history_length=4,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Seat','{ENV_REGEX_NS}/RestFloor']))
    return cfg

def metrics(raw):
    m,d,f=original_metrics(raw)
    env=diag.active_env
    if env is not None:
        due=m['failure']&env.alive&(env.warming==0)
        for i in due.nonzero().flatten().tolist():
            pairs={}
            for side in ('left','right'):
                history=raw.scene[f'diagnostic_{side}_hip'].data.normal_force_matrix_w_history.torch[i]
                values=history.sum(1).norm(dim=-1).max(0).values
                pairs[side]=dict(seat_peak_N=float(values[0]),floor_peak_N=float(values[1]))
            with (env.out/'contact_pairs.jsonl').open('a') as out:
                out.write(json.dumps(dict(case=i,t_s=(env.t+1)*.02,pairs=pairs))+'\n')
    return m,d,f

base.make_cfg=make_cfg
base.metrics=metrics
if __name__=='__main__':base.main()
