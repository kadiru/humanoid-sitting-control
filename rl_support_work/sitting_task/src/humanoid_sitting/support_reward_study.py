"""Training-only reward experiment; no labels are exposed to the policy.

The legacy arm is exactly the existing reward. The gentle arm replaces the
time-integrated impact cost with an episode peak cost and makes leg economy
sensitive to Nm differences during verified, upright, supported sitting.
"""
import torch
from .support_objective import support_objective


class SittingRewardStudy:
    def __init__(self, arm):
        assert arm in ('legacy', 'gentle')
        self.arm = arm
        self.peak_excess2 = None
        self.totals = {}

    def reset(self, ids=None):
        if ids is None:
            self.peak_excess2 = None
        elif self.peak_excess2 is not None:
            self.peak_excess2[ids] = 0.

    def __call__(self, m, issued_delta, residual, alive, dt=.02):
        reward, good, terms = support_objective(m, issued_delta, residual, alive, dt)
        if self.arm == 'gentle':
            if self.peak_excess2 is None:
                self.peak_excess2 = torch.zeros_like(reward)
            excess2 = (m['peak_force_ratio'] - 1.).clamp_min(0).square()
            new_peak = torch.where(alive, torch.maximum(self.peak_excess2, excess2), self.peak_excess2)
            peak_charge = -12. * (new_peak - self.peak_excess2)
            self.peak_excess2 = new_peak.detach()
            economy = 4. * torch.exp(-(m['leg_rms_nm'] / 10.).square()) * good
            # Replace, rather than double-count, the two original terms.
            reward = reward + torch.where(alive,
                dt * (economy - terms['leg_economy'] - terms['impact']), 0.) + peak_charge
            terms = {**terms, 'leg_economy': economy, 'impact': peak_charge / dt}
        # Diagnostic reward contributions are gated exactly like actual reward.
        for name, value in terms.items():
            self.totals[name] = self.totals.get(name, 0.) + float((value * alive).sum()) * dt
        self.totals['failure'] = self.totals.get('failure', 0.) - 5. * float((m['failure'] & alive).sum())
        self.totals['actual_total'] = self.totals.get('actual_total', 0.) + float(reward.sum())
        return reward, good, terms

    def pop_totals(self):
        result, self.totals = self.totals, {}
        return result
