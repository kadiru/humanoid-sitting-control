"""Independent synthetic kinematics and actuator-contract tests; no robot replay."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from continuation import FixedFeet,ClosureFailure
from control_path import encode,assemble,rolling_root,bind_cycle_stages
from interior_audit import audit as interior_audit


class TestModel:
    def __init__(self):
        self.names=[];self.joints=[];self.frames={}
        for side,sign in (('left',1),('right',-1)):
            parent='pelvis'
            for name,axis,p in [('hip_pitch','Y',[0,sign*.1,0]),('hip_roll','X',[0,0,-.04]),
                 ('hip_yaw','Z',[0,0,0]),('knee','Y',[.03,0,-.3]),
                 ('ankle_pitch','Y',[0,0,-.3]),('ankle_roll','X',[.01,0,-.02])]:
                joint=side+'_'+name+'_joint';body=side+'_'+name+'_link'
                self.names.append(joint);self.joints.append(dict(name=joint,axis=axis,body0=[parent],body1=[body]))
                A=np.eye(4);A[:3,3]=p;self.frames[joint]=(A,np.eye(4));parent=body
        self.names.extend('fixed_'+str(i) for i in range(25))
    def fk(self,q):
        poses={'pelvis':np.eye(4)}
        for i,j in enumerate(self.joints):
            M=np.eye(4);M[:3,:3]=Rotation.from_euler(j['axis'].lower(),q[i]).as_matrix()
            poses[j['body1'][0]]=poses[j['body0'][0]]@self.frames[j['name']][0]@M
        return poses


def setup():
    model=TestModel();q=np.zeros(37)
    q[:12]=[-.4,.02,.01,.8,-.4,-.02]*2
    bodies=['pelvis']+[j['body1'][0] for j in model.joints]
    inertia=dict(body_names=bodies,mass=[10.]+[1.]*12,local_com=[[0,0,-.04]]*13)
    f=FixedFeet(model,inertia,q,np.eye(4),np.full(23,-2.),np.full(23,2.))
    return f,q


class ClosureTests(unittest.TestCase):
    def test_analytic_jacobian_matches_independent_differences(self):
        f,q=setup();root=np.eye(4);direction=np.array([1.,0.,-.1])
        _,J,_=f.evaluate(q,root,direction)
        numeric=[]
        for i in range(13):
            samples=[]
            for sign in (-1,1):
                trial=q.copy();r=root.copy()
                if i<12:trial[f.legs[i]]+=sign*1e-6
                else:r[:3,3]+=sign*1e-6*direction
                error,_,com=f.evaluate(trial,r,direction)
                samples.append(np.r_[-error,com[0]])
            numeric.append((samples[1]-samples[0])/2e-6)
        np.testing.assert_allclose(J,np.asarray(numeric).T,rtol=0,atol=2e-9)
    def test_fixed_feet_stay_closed_during_root_motion(self):
        f,q=setup();original=q.copy()
        for k in range(1,21):
            root=np.eye(4);root[:3,3]=[-k*.00002,0.,-k*.00005]
            root[:3,:3]=Rotation.from_euler('y',k*.0001).as_matrix()
            q,root,info=f.solve(q,root)
            self.assertLessEqual(info['maximum_residual'],1e-9)
            for body,anchor in zip(f.feet,f.anchors):
                np.testing.assert_allclose(root@f.model.fk(q)[body],anchor,rtol=0,atol=2e-9)
        np.testing.assert_array_equal(q[12:],original[12:])
    def test_balance_continuation_has_same_foot_and_com_target(self):
        f,q=setup();root=np.eye(4);_,_,com=f.evaluate(q,root)
        root[2,3]=-.001
        q,root,info=f.solve(q,root,direction=[1.,0.,0.],target_com_x=com[0])
        e,_,actual=f.evaluate(q,root)
        self.assertLess(max(abs(e)),1e-9);self.assertAlmostEqual(actual[0],com[0],places=9)
    def test_unreachable_step_returns_failure_without_fallback(self):
        f,q=setup();root=np.eye(4);root[0,3]=1.
        with self.assertRaises(ClosureFailure):f.solve(q,root)
    def test_nonfinite_or_changed_fingers_are_rejected(self):
        f,q=setup()
        for index,value in ((0,float('nan')),(36,.001)):
            trial=q.copy();trial[index]=value
            with self.assertRaises(ClosureFailure):f.solve(trial,np.eye(4))


class CommandTests(unittest.TestCase):
    def nodes(self,n=10):
        return [dict(command=[i*.001]*23+[0.]*14,phase='landing',load=.2,model_effort_ratio=.7)
                for i in range(n+1)]
    def test_both_directions_obey_original_filter(self):
        nodes=self.nodes();p=assemble(nodes,[-2.]*23,[2.]*23,{'synthetic':True})
        for key,rows in (('forward',nodes),('backward',nodes[::-1])):
            current=np.asarray(rows[0]['command'][:23])
            for desired,command in zip(p[key]['desired'],p[key]['commands']):
                current=current+np.clip(.1*(np.asarray(desired)-current),-.003,.003)
                np.testing.assert_allclose(current,command[:23],rtol=0,atol=1e-12)
        self.assertFalse(p['physical_success'])
    def test_speed_and_margin_failures_are_not_resampled(self):
        n=self.nodes();n[1]['command'][0]=.004
        with self.assertRaisesRegex(ValueError,'SPEED'):encode(n,[-2.]*23,[2.]*23)
        with self.assertRaisesRegex(ValueError,'BOX'):encode(self.nodes(),[-2.]*23,[.035]*23)
    def test_time_window_and_fingers_are_preserved(self):
        with self.assertRaisesRegex(ValueError,'TIME'):encode(self.nodes(645),[-2.]*23,[2.]*23)
        n=self.nodes();n[1]['command'][-1]=.001
        with self.assertRaisesRegex(ValueError,'FINGERS'):encode(n,[-2.]*23,[2.]*23)
    def test_stage_binding_preserves_numeric_trajectory_and_qualifies_real_load(self):
        import copy
        nodes=self.nodes(5)
        for node,gap,load in zip(nodes,[.05,.012,0,0,0,0],[0,0,0,.02,.10,.4]):
            node.update(gap=gap,load=load,normal=False)
        p=assemble(nodes,[-2.]*23,[2.]*23,{})
        before=copy.deepcopy(p['forward'])
        bind_cycle_stages(p,1,2)
        self.assertEqual(p['forward'],before)
        self.assertEqual(p['cycle_boundaries'],dict(hover_node=1,touch_node=2,loaded_node=4,release_node=4))
        self.assertTrue(p['nodes'][1]['normal']);self.assertTrue(p['return_nodes'][4]['normal'])
    def test_zero_load_or_wrong_hover_cannot_be_bound(self):
        nodes=self.nodes(4)
        for n in nodes:n.update(gap=0.,load=0.)
        p=assemble(nodes,[-2.]*23,[2.]*23,{})
        with self.assertRaisesRegex(ValueError,'GEOMETRY'):bind_cycle_stages(p,1,2)
        nodes[1]['gap']=.012
        with self.assertRaisesRegex(ValueError,'LOAD'):bind_cycle_stages(p,1,2)


class ContactGeometryTests(unittest.TestCase):
    def test_rolling_translation_keeps_the_material_point_fixed(self):
        class Pivot:
            vertex=np.array([.1,0.,-.1])
            def rotation(self,theta):return Rotation.from_euler('y',theta).as_matrix()
            def translation(self,theta,initial):return initial+self.vertex-self.rotation(theta)@self.vertex
        p=Pivot();touch=.3;position=np.array([-.2,0.,.1])
        fixed=position+p.rotation(touch)@p.vertex
        for theta in np.linspace(0,touch,17):
            root=rolling_root(p,theta,touch,position)
            np.testing.assert_allclose(root[:3,:3]@p.vertex+root[:3,3],fixed,rtol=0,atol=1e-15)
        np.testing.assert_array_equal(rolling_root(p,touch,touch,position)[:3,3],position)
    def test_quarter_control_check_rejects_valid_endpoints_with_bad_interior(self):
        class Chain:
            def fk(self,q):
                T=np.eye(4);T[:3,:3]=Rotation.from_euler('z',-q[0]).as_matrix();T[:3,3]=[1,0,0]
                return {'left':T,'right':T}
        nodes=[]
        for theta in (0.,np.pi/2):
            root=np.eye(4);root[:3,:3]=Rotation.from_euler('z',theta).as_matrix()
            root[:3,3]=[-np.cos(theta),-np.sin(theta),0.]
            nodes.append(dict(q=[theta]+[0.]*36,root=root))
        r=interior_audit(Chain(),nodes,['left','right'],[np.eye(4),np.eye(4)],[[0,0,1]],[0,0,1],0.)
        self.assertFalse(r['passed']);self.assertEqual(r['samples'],4)
        self.assertLess(r['rows'][-1]['foot_error'],1e-12)
        self.assertGreater(r['maximum_foot_error'],.2)


if __name__=='__main__':unittest.main()
