"""Calibrate fore-aft compliance and check the extra travel against the recorded contact."""
from contact_corridor_path import *
def calibrate():
    path = P/'loaded_corner_16/evaluation'
    audit = read(path/'independent_audit.json')
    assert audit['passed'] and sha(path/'trajectory.npz') == audit['files']['trajectory.npz']
    record = np.load(path/'trajectory.npz')
    j = 14
    assert read(path.parent/'bank/geometry.json')[j]['name'] == 'Monarch'
    frame = 725
    normal = record['physical_normal_w'][680:825,j,0].mean(0)
    normal /= np.linalg.norm(normal)
    tangent = np.array([1.,0.,0.]) - normal*normal[0]
    tangent /= np.linalg.norm(tangent)
    pose = record['body'][frame,j].astype(float)
    meta = read(path/'results.json')
    pose[:,:3] -= meta['origins'][j]
    feet = pose[[meta['body_names'].index(n) for n in feet_names]]
    rotation = Rot.from_quat(pose[0,3:]).as_matrix()
    seat = Seat(geoms[30])
    displacements = np.linspace(0., -.002, 17)
    gaps = []
    for displacement in displacements:
        root = pose[0,:3] + tangent * displacement
        joints, error = fit(record['q'][frame,j].copy(), root, rotation, feet)
        assert error < .00002
        gaps.append(measure(robot.state(joints, rotation, root), seat))
    assert np.min(np.array(gaps)[:,2]) > .0006
    matrix = np.load(P/'measured_corner_path_30.npz')['motion'][594,84:].reshape(12,6)
    ids = legs.ravel()
    stiffness = matrix.T @ np.diag(kp[ids]) @ matrix
    direction = np.r_[tangent, 0.,0.,0.]
    compliance = float(direction @ np.linalg.solve(stiffness, direction))
    np.savez(P/'corner_traction_fit.npz', tangent=tangent, normal=normal, ids=ids,
             gain=matrix@direction, compliance=compliance, displacements=displacements,
             gaps=np.array(gaps))
    print('Fore-aft compliance m/N', compliance, 'min other gap', np.min(np.array(gaps)[:,2]))
if __name__ == '__main__':
    calibrate()
