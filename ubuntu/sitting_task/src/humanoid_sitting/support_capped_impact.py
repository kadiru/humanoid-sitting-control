"""Bounded training-only impact objective and exact supported-hold timer."""
import torch
from .support_objective import support_objective


class SupportedHoldTicks:
    def __init__(self):
        self.current = None
        self.peak = None

    def reset(self, ids=None):
        if ids is None:
            self.current = self.peak = None
        elif self.current is not None:
            self.current[ids] = 0
            self.peak[ids] = 0

    def update(self, good):
        if self.current is None:
            self.current = torch.zeros_like(good, dtype=torch.int64)
            self.peak = self.current.clone()
        self.current = torch.where(good, self.current + 1, 0)
        self.peak = torch.maximum(self.peak, self.current)

    def count(self, i):
        return 0 if self.peak is None else int(self.peak[i])


class CappedImpactStudy:
    def __init__(self, arm):
        assert arm in ('legacy', 'capped')
        self.arm = arm
        self.peak = None
        self.hold = SupportedHoldTicks()
        self.totals = {}

    def reset(self, ids=None):
        self.hold.reset(ids)
        if ids is None:
            self.peak = None
        elif self.peak is not None:
            self.peak[ids] = 0.

    def __call__(self, m, issued_delta, residual, alive, dt=.02):
        reward, good, terms = support_objective(m, issued_delta, residual, alive, dt)
        self.hold.update(good)
        if self.arm == 'capped':
            if self.peak is None:
                self.peak = torch.zeros_like(reward)
            cost = (m['peak_force_ratio']-1.).clamp_min(0).square().clamp_max(1.)
            new_peak = torch.where(alive, torch.maximum(self.peak, cost), self.peak)
            charge = -5. * (new_peak-self.peak)
            self.peak = new_peak.detach()
            reward = reward - torch.where(alive, dt*terms['impact'], 0.) + charge
            terms = {**terms, 'impact': charge/dt}
        for name, value in terms.items():
            self.totals[name] = self.totals.get(name, 0.) + float((value*alive).sum())*dt
        self.totals['failure'] = self.totals.get('failure', 0.) - 5.*float((m['failure'] & alive).sum())
        self.totals['actual_total'] = self.totals.get('actual_total', 0.) + float(reward.sum())
        return reward, good, terms

    def pop_totals(self):
        result, self.totals = self.totals, {}
        return result
