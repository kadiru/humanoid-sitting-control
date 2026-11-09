import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sitting_task/src'))
from humanoid_sitting.supported_surface import surface_frame,surface_coordinates,signed_clearance,tangential_velocity,fit_observed_planes

class Tests(unittest.TestCase):
    def test_pitch_yaw_and_local_coordinates(self):
        R=surface_frame(90,90);np.testing.assert_allclose(R[:,2],[0,1,0],atol=1e-12)
        center=np.array([.2,.3,.4]);local=np.array([[.1,-.2,.03]])
        np.testing.assert_allclose(surface_coordinates(local@R.T+center,center,R),local,atol=1e-12)
        np.testing.assert_allclose(signed_clearance(local@R.T+center,R[:,2],center),[.03],atol=1e-12)

    def test_plane_tangent_slip_removes_normal_velocity(self):
        n=surface_frame(65)[:,2];v=4*n+np.array([0,.2,0])
        np.testing.assert_allclose(tangential_velocity(v,np.zeros(3),np.zeros(3),np.zeros(3),n),[0,.2,0],atol=1e-12)

    def test_arbitrary_depth_planes_survive_fitting(self):
        u,v=np.meshgrid(np.linspace(-.2,.2,25),np.linspace(-.25,.25,25));local=np.c_[u.ravel(),v.ravel(),np.zeros(u.size)]
        for angle in [0,20,45,75,90]:
            R=surface_frame(angle);center=np.array([-.4,0,-.25]);pts=local@R.T+center
            planes=fit_observed_planes(pts,np.broadcast_to(R[:,2],pts.shape))
            self.assertEqual(len(planes),1)
            self.assertAlmostEqual(abs(planes[0]['normal']@R[:,2]),1,places=10)
            self.assertLess(planes[0]['rms'],1e-10)

    def test_floor_and_wall_are_not_merged(self):
        u,v=np.meshgrid(np.linspace(-.2,.2,20),np.linspace(-.2,.2,20));flat=np.c_[u.ravel(),v.ravel(),np.zeros(u.size)]
        points=[];normals=[]
        for angle,center in [(0,[0,0,-.7]),(90,[-.4,0,-.1])]:
            R=surface_frame(angle);points.append(flat@R.T+center);normals.append(np.broadcast_to(R[:,2],flat.shape))
        self.assertEqual(len(fit_observed_planes(np.concatenate(points),np.concatenate(normals))),2)



def test_tracker_translation_expiry_and_reset():
    from humanoid_sitting.supported_surface import SupportTracker
    class Model:
        vertices={name:np.array([[0.,0.,0.]]) for name in ['pelvis','torso_link','logo_link']}
        def fk(self,q):
            result={}
            for body,y in [('left_ankle_roll_link',.1),('right_ankle_roll_link',-.1),('torso_link',0.),('logo_link',0.)]:
                T=np.eye(4);T[:3,3]=[0,y,-.7 if 'ankle' in body else .2];T[:3,3]+=q;result[body]=T
            return result
    tracker=SupportTracker(Model());v=np.array([.3,.5,0,0,1,.3,0,0,-.3,.5,.8,1,0],np.float32)
    tracker.update(np.zeros(3),[0,0,-1],[0,0,0],measurement=v)
    out=tracker.update(np.array([0,0,.02]),[0,0,-1],[0,0,0])
    assert abs(out[5]-.28)<1e-6 and abs(out[8]+.28)<1e-6 and out[11]
    for _ in range(26):out=tracker.update(np.array([0,0,.02]),[0,0,-1],[0,0,0])
    assert not out[11]
    tracker.reset();assert not tracker.update(np.zeros(3),[0,0,-1],[0,0,0])[11]

Tests.test_tracker_translation_expiry_and_reset=lambda self:test_tracker_translation_expiry_and_reset()
if __name__=='__main__':unittest.main()
