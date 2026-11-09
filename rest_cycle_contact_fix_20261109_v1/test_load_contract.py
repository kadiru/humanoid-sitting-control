"""Connect actual producer uncertainty to the unchanged release consumer."""
from dataclasses import replace
import math
import unittest
from load_contract import estimate_interval,release_compatibility,validation_verdict
from test_cycle_executor import Harness,measurement


def with_producer_interval(sample):
    estimate=.4 if sample.load_lower_bw and sample.load_lower_bw>0 else 0.
    lower,upper=estimate_interval(estimate,available=True)
    return replace(sample,load_lower_bw=lower,load_upper_bw=upper)


def demonstrate_mismatch():
    """Toy path, constant exact mock estimates; no saved path or robot physics."""
    h=Harness();release_seen=False
    for _ in range(2001):
        row=h.tick(with_producer_interval)
        release_seen |= row['reason']=='await_sensor_qualified_release'
        if row['aborted']:break
    return dict(terminated=row['aborted'],reason=row['reason'],phase=row['phase'],
        path_index=row['path_index'],release_node=h.executor.plan.release.command_count,
        release_wait_observed=release_seen,quiet_samples=row['quiet_samples'],
        sensor_cycle_complete=row['sensor_cycle_complete'],calls=len(h.rows),
        physics_steps=0,synthetic_only=True,physical_success=False)


class ProducerConsumerTests(unittest.TestCase):
    def test_zero_estimate_preserves_original_uncertainty(self):
        self.assertEqual(estimate_interval(0.,available=True),(-.02,.02))
        h=Harness();sample=with_producer_interval(measurement(h.executor,0))
        self.assertFalse(sample.released())

    def test_exact_bound_contradiction_prevents_any_release_estimate(self):
        report=release_compatibility()
        self.assertFalse(report['passed'])
        self.assertGreaterEqual(report['lower_estimate_bound_inclusive_bw'],report['upper_estimate_bound_exclusive_bw'])
        self.assertEqual(report['reason'],'NO_ESTIMATE_CAN_SATISFY_RELEASE_INTERVAL')

    def test_positive_load_qualification_still_works(self):
        h=Harness();sample=measurement(h.executor,0)
        lo,hi=estimate_interval(.4,available=True)
        self.assertTrue(replace(sample,load_lower_bw=lo,load_upper_bw=hi).packet(0.).loaded())

    def test_unknown_and_impossible_estimates_are_not_zero_load(self):
        for value in [None,math.nan,math.inf,-math.inf,-1.]:
            self.assertEqual(estimate_interval(value,available=True),(None,None))
        self.assertEqual(estimate_interval(0.,available=False),(None,None))

    def test_real_producer_contract_stops_at_release_with_original_deadline(self):
        result=demonstrate_mismatch()
        self.assertTrue(result['terminated']);self.assertTrue(result['release_wait_observed'])
        self.assertEqual(result['reason'],'original20s_return_deadline')
        self.assertEqual(result['path_index'],result['release_node'])
        self.assertEqual(result['quiet_samples'],0)
        self.assertFalse(result['sensor_cycle_complete']);self.assertEqual(result['calls'],2001)

    def test_idealized_filter_pass_cannot_be_an_overall_compatibility_pass(self):
        report=validation_verdict(actuator_filter_passed=True,idealized_protocol_passed=True)
        self.assertTrue(report['actuator_filter_passed']);self.assertTrue(report['idealized_sensor_protocol_passed'])
        self.assertFalse(report['passed']);self.assertFalse(report['producer_to_controller_compatible'])


if __name__=='__main__':unittest.main()
