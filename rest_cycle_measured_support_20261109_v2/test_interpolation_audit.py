"""Analytic closed-chain example: valid endpoints can have an invalid interior."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from interpolation_audit import audit_interpolation


class ClosedChain:
    def fk(self,q):
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('z',-q[0]).as_matrix()
        pose[:3,3]=[1,0,-1]
        return {'left':pose,'right':pose}


def node(theta):
    root=np.eye(4);root[:3,:3]=Rotation.from_euler('z',theta).as_matrix()
    root[:3,3]=[-np.cos(theta),-np.sin(theta),1.]
    q=[theta]+[0.]*36
    return dict(root=root.tolist(),q=q,command=q)


class InterpolationChecks(unittest.TestCase):
    def run_path(self,end):
        a,b=node(0),node(end)
        commands=np.linspace(a['command'],b['command'],601)[1:]
        return audit_interpolation(ClosedChain(),[a,b],dict(commands=commands),
            ['left','right'],[np.eye(4),np.eye(4)],[[0,0,0]],[0,0,1],0.)
    def test_exact_endpoints_do_not_hide_interior_foot_motion(self):
        r=self.run_path(np.pi/2)
        self.assertFalse(r['passed']);self.assertLess(r['maximum_endpoint_foot_error'],1e-12)
        self.assertGreater(r['maximum_foot_error'],.20)
        self.assertFalse(r['first_violation']['endpoint'])
    def test_stationary_closed_chain_passes_without_physical_claim(self):
        # Distinct tiny targets keep command-node association unambiguous.
        r=self.run_path(1e-6)
        self.assertTrue(r['passed']);self.assertFalse(r['physical_success'])
        self.assertEqual(r['control_samples'],600)


if __name__=='__main__':unittest.main()
