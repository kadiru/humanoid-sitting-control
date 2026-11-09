"""Offline evaluation contracts for unavailable startup observations."""
import unittest
from analyze_shared_frame_six import assess, measurement_quality
from evaluate_shared_frame import metric_keys


class EvaluationTests(unittest.TestCase):
    def test_later_seat_observation_is_included_in_metric_inventory(self):
        keys=metric_keys([dict(pelvis_position_error_cm=.1),
                          dict(pelvis_position_error_cm=.2,clearance_error_cm=.3)])
        self.assertIn('clearance_error_cm',keys)

    def test_unobserved_startup_remains_in_availability_denominator(self):
        first=dict(case=0,frame=0,stamp=0.,status='no_seat',valid=False)
        second=dict(case=0,frame=1,stamp=.1,status='visible',valid=True,direct_clearance_m=.1,
            quality=dict(alignment_valid=True,quantities={q:dict(valid=True,reasons=[])
                for q in ('clearance','forward','lateral','yaw')}))
        errors=[dict(case=0,frame=0,truth_clearance_m=.2),
            dict(case=0,frame=1,truth_clearance_m=.1,forward_offset_error_cm=.2,
                 lateral_offset_error_cm=.3,yaw_axis_error_deg=.1)]
        result=assess([first,second],errors)
        self.assertEqual(result['samples'],2)
        self.assertEqual(result['alignment_valid'],1)
        for q in result['quantities'].values():
            self.assertEqual(q['available'],1);self.assertEqual(q['fraction'],.5)
            self.assertEqual(q['invalid_reasons'],{'no_seat_reference':1})
        self.assertEqual(result['violations'],[])
        self.assertNotIn('quality',first)

    def test_missing_quality_cannot_hide_a_claimed_measurement(self):
        with self.assertRaises(ValueError):
            measurement_quality(dict(valid=True,status='no_seat'))


if __name__=='__main__':
    unittest.main(verbosity=2)
