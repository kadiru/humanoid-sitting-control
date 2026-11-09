"""Manager environment with measured sitting episode statistics."""
from isaaclab.envs import ManagerBasedRLEnv


class SittingEnv(ManagerBasedRLEnv):
    def __init__(self, *args, **kwargs):
        self.completed_sitting_episodes = []
        super().__init__(*args, **kwargs)

    def _reset_idx(self, env_ids):
        if env_ids is None or isinstance(env_ids, slice):
            import torch
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids if env_ids is not None else slice(None)]
        records = []
        if hasattr(self, "sit_hold"):
            for i in env_ids.tolist():
                if self.episode_length_buf[i] > 0:
                    records.append({"success": bool(self.sit_hit[i]),
                                    "peak_hold_s": float(self.sit_peak_hold[i]),
                                    "peak_support_fraction": float(self.sit_peak_support[i]),
                                    "duration_s": float(self.episode_length_buf[i]) * self.step_dt})
                    if getattr(self.cfg, "require_clean_seat_contacts", False):
                        sensor=self.scene["all_contacts"]
                        forces=sensor.data.net_normal_forces_w.torch[i]
                        records[-1]["terminal_contacts_N"]={n:v.tolist() for n,v in zip(sensor.body_names,forces) if float(v.norm())>10.0}
                        records[-1]["terminal_seat_forces_N"]={f"{side}_{body}":self.scene[f"{side}_{body}_seat"].data.normal_force_matrix_w.torch[i].sum(dim=(0,1)).tolist() for side in ("left","right") for body in ("foot","knee")}
                        records[-1]["terminal_root_xyz"]=(self.scene["robot"].data.root_pos_w.torch[i]-self.scene.env_origins[i]).tolist()
            self.completed_sitting_episodes.extend(records)
        super()._reset_idx(env_ids)
        if hasattr(self, "sit_hold"):
            self.sit_hold[env_ids] = 0
            self.sit_peak_hold[env_ids] = 0
            self.sit_peak_support[env_ids] = 0
            self.sit_hit[env_ids] = False
            self.sit_descent_ok[env_ids] = True
        if records:
            self.extras["log"]["Sitting/success_rate"] = sum(r["success"] for r in records) / len(records)
            self.extras["log"]["Sitting/peak_hold_s"] = sum(r["peak_hold_s"] for r in records) / len(records)
