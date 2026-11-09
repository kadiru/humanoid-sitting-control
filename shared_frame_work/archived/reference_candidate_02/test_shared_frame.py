"""CPU contract/regression tests; no simulator needed."""
import inspect
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import (transform, apply, level_rotation, register,
    support_patch, SharedFrameTracker, SensorFrame, unproject, depth_edges, regularize_plane_normals,
    DirectionalBudget, boundary_evidence, task_quality, reacquire, bounded_map, support_junctions,
    plane_clearance_allowance, associate_seat_plane, AnchoredReferenceBank,
    conditioned_reference_match)
import humanoid_sitting.shared_frame as module


class FrameTests(unittest.TestCase):
    def test_transform_direction_and_composition(self):
        A = transform(Rotation.from_euler('xyz', [.2, -.3, 1.1]).as_matrix(), [.8, -.3, .7])
        B = transform(Rotation.from_euler('z', -.5).as_matrix(), [-.4, .2, .3])
        p = np.array([[.2, .3, .4]])
        np.testing.assert_allclose(apply(A@B, p), apply(A, apply(B, p)), atol=1e-12)
        np.testing.assert_allclose(apply(np.linalg.inv(A), apply(A, p)), p, atol=1e-12)

    def test_gravity_and_optical_axes(self):
        R = Rotation.from_euler('xy', [.2, -.3]).as_matrix()
        g = R.T @ [0, 0, -1]
        np.testing.assert_allclose(level_rotation(g) @ g, [0, 0, -1], atol=1e-12)
        K = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        p = unproject(np.array([[2., 2.]]), K, np.eye(4))
        np.testing.assert_allclose(p, [[0, 0, 2], [2, 0, 2]])

    def test_observable_registration_with_yaw_and_translation(self):
        rng = np.random.default_rng(41017)
        p = rng.uniform(-.7, .7, (4000, 3)); n = np.zeros_like(p)
        for i, axis in enumerate([0, 1, 2]):
            sl = slice(i*1300, (i+1)*1300)
            p[sl, axis] = [-.6, .6, 0][i]; n[sl, axis] = 1
        p, n = p[:3900], n[:3900]
        actual = transform(Rotation.from_euler('z', .025).as_matrix(), [.017, -.012, .025])
        source = apply(np.linalg.inv(actual), p)
        got, diag = register(source, p, n, np.eye(4), iterations=25)
        self.assertTrue(diag['accepted']); self.assertEqual(diag['rank'], 4)
        np.testing.assert_allclose(got, actual, atol=1e-5)

    def test_floor_cannot_measure_inplane_motion(self):
        x, y = np.mgrid[-1:1:40j, -1:1:40j]
        p = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        n = np.tile([0, 0, 1.], (len(p), 1))
        _, diag = register(p, p, n, transform(p=[.01, .02, 0]))
        self.assertLess(diag['rank'], 4); self.assertFalse(diag['accepted'])

    def test_depth_edges_exclude_image_boundary(self):
        K = np.array([[100., 0, 5], [0, 100., 5], [0, 0, 1.]])
        d = np.ones((10, 10))
        self.assertEqual(len(depth_edges(d, K, np.eye(4))), 0)
        d[:, 5:] = 1.5
        edge = depth_edges(d, K, np.eye(4))
        self.assertGreater(len(edge), 0)
        np.testing.assert_allclose(edge[:, 0], -.01)
        np.testing.assert_allclose(edge[:, 2], 1.)
        d[:] = np.inf; d[0, 0] = np.nan
        with np.errstate(invalid='raise'):
            self.assertEqual(len(depth_edges(d, K, np.eye(4))), 0)

    def test_contour_restores_planar_scene_observability(self):
        x, y = np.mgrid[-1:1:40j, -1:1:40j]
        p = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        n = np.tile([0, 0, 1.], (len(p), 1))
        a = np.linspace(0, 2*np.pi, 100, endpoint=False)
        boundary = np.c_[.3*np.cos(a), .5*np.sin(a), np.full(len(a), .3)]
        actual = transform(Rotation.from_euler('z', .03).as_matrix(), [.02, -.01, .01])
        got, diag = register(apply(np.linalg.inv(actual), p), p, n, np.eye(4),
            edges=apply(np.linalg.inv(actual), boundary), target_edges=boundary, iterations=25)
        self.assertEqual(diag['rank'], 4)
        np.testing.assert_allclose(got, actual, atol=1.e-4)

    def test_depth_noise_does_not_create_floor_xy_information(self):
        rng = np.random.default_rng(12)
        p = rng.uniform(-1, 1, (2000, 3)); p[:, 2] = rng.normal(0, .005, len(p))
        normals = np.tile([0., 0., 1.], (len(p), 1))+rng.normal(0, .035, p.shape)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals, _ = regularize_plane_normals(p, normals)
        _, diag = register(p, p, normals, np.eye(4))
        self.assertEqual(diag['rank'], 1)
        self.assertFalse(diag['accepted'])

    def test_hole_is_not_usable_support(self):
        x, y = np.mgrid[-.3:.3:.0125, -.45:.45:.0125]
        p = np.c_[x.ravel(), y.ravel(), np.full(x.size, .3)]
        p = p[~((abs(p[:, 0]) < .08) & (abs(p[:, 1]) < .08))]
        patch = support_patch(dict(points=p, normal=np.array([0., 0., 1.]), offset=-.3))
        self.assertIsNotNone(patch)
        self.assertFalse(np.any(np.linalg.norm(patch['safe_cells'], axis=1) < .05))

    def test_curved_support_axis_is_not_polygon_facet(self):
        x, y = np.mgrid[-.25:.25:.009, -.46:.46:.009]
        xy = np.c_[x.ravel(), y.ravel()]
        xy = xy[(xy[:, 0]/.25)**2+(xy[:, 1]/.46)**2 < 1.]
        yaw = .31
        xy = xy @ Rotation.from_euler('z', yaw).as_matrix()[:2, :2].T
        a = np.linspace(0, 2*np.pi, 200, endpoint=False)
        contour = np.c_[.25*np.cos(a), .46*np.sin(a)] @ Rotation.from_euler('z', yaw).as_matrix()[:2, :2].T
        patch = support_patch(dict(points=np.c_[xy, np.full(len(xy), .3)], normal=np.array([0.,0.,1.]), offset=-.3),
                              contour=np.c_[contour, np.full(len(contour), .3)])
        est = np.arctan2(patch['T'][1, 0], patch['T'][0, 0])
        error = (est-yaw+np.pi/2)%np.pi-np.pi/2
        self.assertLess(abs(error), np.deg2rad(2.))
        self.assertEqual(patch['axis_method'], 'fitted_observed_contour_ellipse')

    def test_rectangle_is_not_promoted_to_curved_model(self):
        from humanoid_sitting.shared_frame import fit_curved_contour
        x = np.linspace(-.25, .25, 80); y = np.linspace(-.45, .45, 80)
        edge = np.r_[np.c_[x, np.full_like(x, -.45)], np.c_[x, np.full_like(x, .45)],
                     np.c_[np.full_like(y, -.25), y], np.c_[np.full_like(y, .25), y]]
        self.assertIsNone(fit_curved_contour(edge, np.zeros(2), np.array([.5, .9]), 0.))

    def test_disconnected_same_height_edges_do_not_change_support_axis(self):
        x,y = np.mgrid[-.25:.25:.009, -.46:.46:.009]
        xy = np.c_[x.ravel(), y.ravel()]
        xy = xy[(xy[:,0]/.25)**2+(xy[:,1]/.46)**2 < 1.]
        yaw = -.11; R = Rotation.from_euler('z', yaw).as_matrix()[:2,:2]
        xy = xy @ R.T
        angle = np.linspace(0,2*np.pi,200,endpoint=False)
        boundary = np.c_[.25*np.cos(angle),.46*np.sin(angle)] @ R.T
        # Separate objects can share a height with the seat but not its extent.
        foreign = np.c_[np.full(16,.55),np.linspace(-.18,.18,16)]
        contour = np.c_[np.r_[boundary,foreign],np.full(216,.3)]
        plane = dict(points=np.c_[xy,np.full(len(xy),.3)],normal=np.array([0.,0.,1.]),offset=-.3)
        patch = support_patch(plane,contour=contour)
        self.assertEqual(patch['axis_method'],'fitted_observed_contour_ellipse')
        angle_error = (np.arctan2(patch['T'][1,0],patch['T'][0,0])-yaw+np.pi/2)%np.pi-np.pi/2
        self.assertLess(abs(angle_error),np.deg2rad(.5))
        np.testing.assert_allclose(patch['T'][:2,3],[0,0],atol=.002)

    def test_parallel_edges_do_not_define_an_ellipse(self):
        from humanoid_sitting.shared_frame import fit_curved_contour
        rng = np.random.default_rng(2)
        x = np.linspace(-.18, .18, 150)
        edge = np.r_[np.c_[x, np.full_like(x, -.44)], np.c_[x, np.full_like(x, .44)]]
        edge += rng.normal(0, .002, edge.shape)
        self.assertIsNone(fit_curved_contour(edge, np.zeros(2), np.array([.4, .88]), 0.))

    def test_reject_skew_delay_and_reordered_frames(self):
        tracker = SharedFrameTracker(None, np.eye(4))
        kwargs = dict(stamp=1., received=1.04, depth=np.ones((2, 2)), K=np.eye(3),
                      q=np.zeros(1), gravity=np.array([0, 0, -1]), delta_rotation=np.eye(3),
                      encoder_stamp=1., imu_stamp=1.)
        with self.assertRaisesRegex(ValueError, 'skew'):
            tracker.update(SensorFrame(**(kwargs | dict(encoder_stamp=.98))))
        with self.assertRaisesRegex(ValueError, 'delay'):
            tracker.update(SensorFrame(**(kwargs | dict(received=1.4))))
        tracker.stamp = 1.; tracker.epoch = 0
        with self.assertRaisesRegex(ValueError, 'Out-of-order'):
            tracker.update(SensorFrame(**kwargs))
        tracker.reset(); self.assertIsNone(tracker.T); self.assertIsNone(tracker.seat)

    def test_runtime_has_no_privileged_api(self):
        source = inspect.getsource(module)
        for token in ('root_pos_w', 'root_pose_w', 'body_pose_w', 'env_origins', 'geometry_bank',
                      'truth.npz', 'isaaclab', 'contact_force'):
            self.assertNotIn(token, source)

    def test_directional_budget_does_not_invalidate_observed_height(self):
        budget = DirectionalBudget()
        diag = dict(accepted=True, observed_projector=np.diag([1., 1., 0., 1.]))
        for _ in range(100):
            budget.update(.1, diag, True)
        self.assertLess(budget.allowance([0, 0, 0, 1]), .009)
        self.assertGreater(budget.allowance([0, 0, 1, 0]), .15)
        budget.update(.1, dict(accepted=True, observed_projector=np.eye(4)), True)
        self.assertAlmostEqual(budget.allowance([0, 0, 1, 0]), .008)

    def test_rejected_registration_does_not_reset_budget(self):
        budget = DirectionalBudget()
        budget.update(1., dict(accepted=False, observed_projector=np.eye(4)), True)
        self.assertGreater(budget.allowance([0, 1, 0, 0]), .02)

    def test_cropped_patch_does_not_establish_hidden_center(self):
        x, y = np.mgrid[-.2:.2:35j, -.4:.4:50j]
        xy = np.c_[x.ravel(), y.ravel()]
        ys = np.linspace(-.4, .4, 80)
        contour = np.c_[np.full_like(ys, -.2), ys, np.full_like(ys, .3)]
        evidence = boundary_evidence(xy, contour, transform(p=[0, 0, .3]))
        self.assertFalse(evidence['forward']); self.assertFalse(evidence['lateral'])
        self.assertFalse(evidence['yaw'])

    def test_direct_clearance_is_independent_of_odometry_gauge(self):
        T = transform(Rotation.from_euler('z', .4).as_matrix(), [4., -3., .7])
        seat = dict(T=transform(p=[4., -3., .3]), axis_ambiguous=False,
                    boundary_evidence=dict(forward=False, lateral=False, yaw=True))
        x, y = np.mgrid[3.5:4.5:30j, -3.5:-2.5:30j]
        plane = dict(normal=np.array([0., 0., 1.]), offset=-.3, rms=.002,
                     points=np.c_[x.ravel(), y.ravel(), np.full(x.size, .3)])
        vertices = np.array([[0, 0, -.1], [.1, 0, 0]])
        quality, clearance = task_quality(seat, T, dict(accepted=False), DirectionalBudget(),
                                          0., plane, vertices, True)
        self.assertAlmostEqual(clearance, .3)
        self.assertTrue(quality['quantities']['clearance']['valid'])
        self.assertFalse(quality['alignment_valid'])
        quality, clearance = task_quality(seat, T, dict(accepted=False), DirectionalBudget(),
                                          1.1, None, vertices, True)
        self.assertIsNone(clearance)
        self.assertTrue(all(not x['valid'] for x in quality['quantities'].values()))

    def test_reacquire_translation_outside_local_icp_gate(self):
        x, y = np.mgrid[-.8:.8:40j, -.8:.8:40j]
        points = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        normals = np.tile([0., 0., 1.], (len(points), 1))
        a = np.linspace(0, 2*np.pi, 180, endpoint=False)
        edges = np.c_[.3*np.cos(a), .5*np.sin(a), np.full(len(a), .3)]
        actual = transform(p=[.02, .12, 0.])
        T, diag = reacquire(apply(np.linalg.inv(actual), points), points, normals, np.eye(4),
                            apply(np.linalg.inv(actual), edges), edges)
        self.assertTrue(diag['accepted']); self.assertEqual(diag['recovery'], 'reacquired')
        np.testing.assert_allclose(T, actual, atol=1e-4)

    def test_blind_common_slip_is_not_qualified_by_foot_agreement(self):
        seat = dict(T=np.eye(4), axis_ambiguous=False,
                    boundary_evidence=dict(forward=True, lateral=True, yaw=True))
        budget = DirectionalBudget(); budget.update(.1, dict(accepted=False), True)
        quality, _ = task_quality(seat, np.eye(4), dict(accepted=False), budget,
                                  .1, None, np.zeros((1,3)), True)
        self.assertFalse(quality['alignment_valid'])
        self.assertIn('motion_direction_unobserved', quality['quantities']['lateral']['reasons'])

    def test_reacquisition_rejects_a_single_straight_edge(self):
        x,y=np.mgrid[-.8:.8:35j,-.8:.8:35j]
        points=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
        normals=np.tile([0.,0.,1.],(len(points),1))
        edge=np.c_[np.full(150,-.3),np.linspace(-1,1,150),np.full(150,.3)]
        _,diag=reacquire(points,points,normals,np.eye(4),edge,edge)
        self.assertFalse(diag['accepted'])

    def test_map_budget_does_not_crop_the_seat_or_scene_extent(self):
        floor=np.c_[np.linspace(-5,5,5000),np.zeros(5000),np.zeros(5000)]
        seat=np.c_[np.linspace(-.8,-.2,500),np.zeros(500),np.full(500,.3)]
        points=np.r_[floor,seat]; points=points[np.argsort(points[:,0])]
        normals=np.tile([0.,0.,1.],(len(points),1))
        p,n=bounded_map(points,normals,limit=1000)
        self.assertLessEqual(len(p),1000)
        self.assertEqual(np.sum(p[:,2]>.1),len(seat))
        self.assertEqual(p[:,0].min(),-5); self.assertEqual(p[:,0].max(),5)
        np.testing.assert_allclose(n,np.tile([0.,0.,1.],(len(p),1)))

    @staticmethod
    def junction_fixture(gap=False, floating=False, displaced=False):
        x,y=np.mgrid[-.25:.251:.0125,-.45:.451:.0125]
        support=np.c_[x.ravel(),y.ravel()]
        t,h=np.mgrid[-.44:.441:.016,.018:.25:.018]
        wall=np.c_[np.full(t.size,-.60 if displaced else -.25),t.ravel(),h.ravel()+.3]
        if gap: wall=wall[abs(wall[:,1])>.13]
        if floating: wall[:,2]+=.12
        normal=np.tile([1.,0.,0.],(len(wall),1))
        plane=dict(points=np.c_[support,np.full(len(support),.3)],normal=np.array([0.,0.,1.]),offset=-.3)
        return plane,support,wall,normal

    def test_measured_wall_confirms_missing_seat_boundary(self):
        plane,xy,wall,normal=self.junction_fixture()
        t=np.linspace(-.45,.45,80)
        contour=np.c_[np.full_like(t,.25),t,np.full_like(t,.3)]
        before=support_patch(plane,contour=contour)
        after=support_patch(plane,contour=contour,context_points=wall,context_normals=normal)
        self.assertFalse(before['boundary_evidence']['forward'])
        self.assertTrue(after['boundary_evidence']['forward'])
        self.assertEqual(after['boundary_evidence']['junction_count'],1)
        np.testing.assert_allclose(after['junctions'][:,0],-.25,atol=.001)
        np.testing.assert_allclose(after['junctions'][:,2],.3,atol=.001)

    def test_junction_is_rotation_equivariant(self):
        plane,xy,wall,normal=self.junction_fixture()
        R=Rotation.from_euler('z',.37).as_matrix()
        junction,_=support_junctions(plane,xy@R[:2,:2].T,wall@R.T,normal@R.T)
        self.assertGreater(len(junction),10)
        local=junction@R
        np.testing.assert_allclose(local[:,0],-.25,atol=.001)

    def test_floating_wall_does_not_invent_seat_junction(self):
        plane,xy,wall,normal=self.junction_fixture(floating=True)
        junction,evidence=support_junctions(plane,xy,wall,normal)
        self.assertEqual(len(junction),0);self.assertEqual(evidence,[])

    def test_distant_wall_does_not_confirm_cropped_seat_extent(self):
        plane,xy,wall,normal=self.junction_fixture(displaced=True)
        junction,_=support_junctions(plane,xy,wall,normal)
        self.assertEqual(len(junction),0)

    def test_wall_junction_does_not_bridge_unobserved_span(self):
        plane,xy,wall,normal=self.junction_fixture(gap=True)
        junction,_=support_junctions(plane,xy,wall,normal)
        self.assertGreater(len(junction),10)
        self.assertFalse(np.any(abs(junction[:,1])<.10))

    def test_thin_visible_strip_does_not_certify_extrapolated_clearance(self):
        x, y = np.mgrid[-.28:-.25:15j, -.15:.15:30j]
        plane = dict(points=np.c_[x.ravel(), y.ravel(), .3+.15*x.ravel()], rms=.001)
        quality = plane_clearance_allowance(plane, np.array([[0., 0., .2], [.1, .1, .2]]))
        self.assertFalse(quality['conditioned'])

    def test_broad_plane_supports_clearance_with_noise(self):
        x, y = np.mgrid[-.3:.3:40j, -.45:.45:60j]
        plane = dict(points=np.c_[x.ravel(), y.ravel(), np.full(x.size, .3)], rms=.005)
        quality = plane_clearance_allowance(plane, np.array([[0., 0., .2], [.1, .1, .2]]))
        self.assertTrue(quality['conditioned'])
        self.assertLess(quality['allowance_m'], .02)

    def test_collinear_plane_samples_cannot_establish_clearance(self):
        x = np.linspace(-.3, .3, 100)
        plane = dict(points=np.c_[x, np.zeros_like(x), np.full_like(x, .3)], rms=.001)
        self.assertFalse(plane_clearance_allowance(plane, np.array([[0., .1, .2]]))['conditioned'])

    def test_registration_rejects_incompatible_surface_normals(self):
        x, y = np.mgrid[-.5:.5:30j, -.5:.5:30j]
        p = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        target_n = np.tile([0., 0., 1.], (len(p), 1))
        source_n = np.tile([1., 0., 0.], (len(p), 1))
        _, diag = register(p, p, target_n, np.eye(4), source_normals=source_n)
        self.assertFalse(diag['accepted'])
        self.assertEqual(diag['overlap'], 0.)

    def test_lone_floor_cannot_become_seat_after_pose_loss(self):
        x,y = np.mgrid[-.5:.5:30j,-.5:.5:30j]
        plane = dict(points=np.c_[x.ravel(),y.ravel(),np.full(x.size,.3)],
                     height=.3,normal=np.array([0.,0.,1.]),rms=.002)
        selected, evidence = associate_seat_plane([plane], .3, .3, False)
        self.assertIsNone(selected); self.assertFalse(evidence['confirmed'])

    def test_floor_seat_separation_survives_wrong_odometry_height(self):
        x,y = np.mgrid[-.5:.5:30j,-.5:.5:30j]
        def plane(z):
            return dict(points=np.c_[x.ravel(),y.ravel(),np.full(x.size,z)],
                        height=z,normal=np.array([0.,0.,1.]),rms=.002)
        floor,seat=plane(.4),plane(.7)
        selected,evidence=associate_seat_plane([floor,seat],.3,.3,False)
        self.assertIs(selected,seat)
        self.assertEqual(evidence['basis'],'current_floor_seat_separation')

    def test_ambiguous_current_support_planes_are_withheld(self):
        x,y=np.mgrid[-.5:.5:30j,-.5:.5:30j]
        planes=[dict(points=np.c_[x.ravel(),y.ravel(),np.full(x.size,z)],
                     height=z,normal=np.array([0.,0.,1.]),rms=.002) for z in (0.,.295,.305)]
        selected,evidence=associate_seat_plane(planes,.3,.3,False)
        self.assertIsNone(selected);self.assertEqual(evidence['candidates'],2)

    def test_reference_consensus_rejects_single_or_disagreeing_views(self):
        diag = dict(accepted=True, rank=4, rms_m=.002, overlap=.9)
        A = transform(p=[.1, .2, .3]); B = transform(p=[.12, .2, .3])
        for solutions in [[(1., A, diag.copy())], [(1., A, diag.copy()), (2., B, diag.copy())]]:
            pose, result = AnchoredReferenceBank.consensus(solutions, np.eye(4))
            self.assertFalse(result['accepted'])
            np.testing.assert_array_equal(pose, np.eye(4))
        pose, result = AnchoredReferenceBank.consensus(
            [(1., A, diag.copy()), (2., A.copy(), diag.copy())], np.eye(4))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['reference_allowance_m'], .008)
        np.testing.assert_array_equal(pose, A)

    def test_noisy_wall_normals_do_not_certify_weak_reference_direction(self):
        # A floor and one wall do not observe translation along that wall.
        # Small normal-estimation errors can nevertheless cross the basic rank
        # threshold; even perfect residual/overlap is not enough to use a view.
        rng = np.random.default_rng(312)
        p = rng.uniform(-.6, .6, (2000, 3)); p[:1000, 2] = 0.; p[1000:, 0] = -.6
        n = np.zeros_like(p); n[:1000, 2] = 1.; n[1000:, 0] = 1.
        n[1000:, 1] = rng.normal(0., .075, 1000)
        n /= np.linalg.norm(n, axis=1)[:, None]
        _, diagnostic = register(p, p, n, np.eye(4), source_normals=n,
                                  minimum_overlap=.6, maximum_rms=.008)
        self.assertTrue(diagnostic['accepted']); self.assertEqual(diagnostic['rank'], 4)
        self.assertFalse(conditioned_reference_match(diagnostic))

    def test_reference_bank_requires_direct_anchor_and_immutable_cycle(self):
        bank = AnchoredReferenceBank()
        x, y = np.mgrid[-.5:.5:30j, -.5:.5:30j]
        p = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        n = np.tile([0., 0., 1.], (len(p), 1))
        a = np.linspace(0., 2*np.pi, 160, endpoint=False)
        edges = np.c_[.3*np.cos(a), .45*np.sin(a), np.full(len(a), .3)]
        diag = dict(accepted=True, rank=4, reference='direct_anchor_view_consensus')
        anchor = (p.copy(), n.copy(), edges.copy())
        self.assertEqual(bank.consider(1., p, n, edges, np.eye(4), diag, anchor), 'not_directly_anchored')
        self.assertEqual(len(bank.views), 0)
        diag['reference'] = 'immutable_initial_depth_scan'
        self.assertEqual(bank.consider(1., p, n, edges, np.eye(4), diag, anchor), 'admitted_direct_anchor')
        p += [1., 0., 0.]; edges += [1., 0., 0.]
        np.testing.assert_array_equal(bank.views[0]['points'], anchor[0])
        np.testing.assert_array_equal(bank.views[0]['edges'], anchor[2])
        self.assertFalse(bank.views[0]['points'].flags.writeable)
        self.assertEqual(bank.consider(3., p, n, edges, transform(p=[.1,0.,0.]), diag, anchor),
                         'reverse_verification_failed')
        self.assertEqual(len(bank.views), 1)

    def test_reference_switch_retains_extra_registration_allowance(self):
        budget = DirectionalBudget()
        budget.update(.1, dict(accepted=True, observed_projector=np.eye(4), reference_allowance_m=.008), True)
        self.assertAlmostEqual(budget.allowance([0., 1., 0., 0.]), .016)
        self.assertAlmostEqual(budget.allowance([2., 0., 0., 0.]), .032)

    def test_fused_map_cannot_move_immutable_pose_reference(self):
        class Model:
            vertices = {name: np.array([[-.03, -.03, -.02], [.03, .03, .02]]) for name in
                ('pelvis', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_knee_link',
                 'right_knee_link', 'torso_link', 'left_elbow_roll_link', 'right_elbow_roll_link')}
            def fk(self, q):
                return {name: transform(p=[0., 0., -.7 if 'ankle' in name else 0.]) for name in self.vertices}
            def self_mask(self, p, poses): return np.ones(len(p), bool)
            def proxy(self, poses): return self.vertices['pelvis']
            def skeleton(self, poses): return np.array([[[0., 0., 0.], [0., 0., -.7]]])
        mount = transform(np.diag([1., -1., -1.]), [0., 0., .1])
        tracker = SharedFrameTracker(Model(), mount)
        depth = np.ones((80, 100)); depth[20:60, 20:80] = .7
        K = np.array([[80., 0., 50.], [0., 80., 40.], [0., 0., 1.]])
        def frame(t, d):
            return SensorFrame(t, t, d, K, np.zeros(1), np.array([0.,0.,-1.]), np.eye(3), t, t)
        initial = tracker.update(frame(0., depth))
        reference = tracker.reference_points.copy()
        tracker.map_points += [2., 1., 0.]
        tracker.map_edges += [2., 1., 0.]
        result = tracker.update(frame(.1, depth))
        np.testing.assert_allclose(result['T_O_P'], initial['T_O_P'], atol=1.e-6)
        np.testing.assert_array_equal(tracker.reference_points, reference)
        self.assertFalse(tracker.reference_points.flags.writeable)
        before = tracker.map_points.copy()
        blind = tracker.update(frame(.2, np.full_like(depth, np.nan)))
        self.assertFalse(blind['registration']['fusion_allowed'])
        np.testing.assert_array_equal(tracker.map_points, before)
        tracker.reset()
        self.assertIsNone(tracker.reference_points)


if __name__ == '__main__':
    unittest.main(verbosity=2)
