"""Exercise the real audit entry point without running geometry or physics."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from corner_audit import audit
from test_plan_contract import example


class AuditInputGateChecks(unittest.TestCase):
    def setUp(self):
        self.m=SimpleNamespace(lower=np.full(23,-2.),upper=np.full(23,2.),state=Mock(),
                               foot_bodies=['left','right'],feet=[np.eye(4),np.eye(4)],
                               mesh={'pelvis':[[0,0,0]]})
        self.plan=SimpleNamespace(mechanics=self.m,previous=np.zeros(37),x=Mock(return_value=None),
                                  model=Mock(),n=np.array([0,0,1]),d=0.)
    def test_invalid_intermediate_load_never_reaches_mechanics(self):
        p=example();p['nodes'][1]['load']=float('nan')
        result=audit(self.plan,p,None,None,None,None)
        self.assertFalse(result['model_admitted']);self.assertFalse(result['physical_success'])
        self.m.state.assert_not_called();self.plan.x.assert_not_called()
    @patch('corner_audit.audit_interpolation',return_value={'passed':True})
    def test_nonfinite_computed_torque_never_reaches_mesh_checks(self,interpolation):
        fields=dict(foot_error=(12,),com=(3,),tau=(23,),cop=(3,),
                    foot_forces=(2,3),foot_points=(2,3),base_residual=(6,))
        self.m.state.return_value={k:np.zeros(shape) for k,shape in fields.items()}
        self.m.state.return_value['tau'][1]=float('nan')
        result=audit(self.plan,example(),None,None,None,None)
        self.assertFalse(result['model_admitted']);self.assertFalse(result['tests']['finite_mechanics'])
        self.assertEqual(result['invalid_mechanics_node'],0)
    @patch('corner_audit.audit_interpolation',side_effect=[{'passed':True},{'passed':False}])
    def test_failed_return_interpolation_blocks_admission(self,interpolation):
        result=audit(self.plan,example(),None,None,None,None)
        self.assertFalse(result['model_admitted']);self.assertFalse(result['physical_success'])
        self.assertFalse(result['tests']['interpolation'])
        self.assertEqual(interpolation.call_count,2)
        self.m.state.assert_not_called();self.plan.x.assert_not_called()
    @patch('corner_audit.audit_interpolation',side_effect=ValueError('NONFINITE_INTERPOLATED_STATE'))
    def test_unknown_interpolation_cannot_be_admitted(self,interpolation):
        result=audit(self.plan,example(),None,None,None,None)
        self.assertFalse(result['model_admitted']);self.assertFalse(result['physical_success'])
        self.assertIn('NONFINITE',result['interpolation_error'])
        self.m.state.assert_not_called()


if __name__=='__main__':unittest.main()
