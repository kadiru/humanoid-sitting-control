"""Copy current backend state without invoking public-state refresh properties.

Requires nonidentity joint ordering, separate public/backend buffers, unchanged
cache bytes and timestamps around EVERY capture. No captured value enters control.
"""
import numpy as np
import torch
import warp as wp

def tensor(x):
    return wp.to_torch(x)

def copy(x):
    return tensor(x).detach().cpu().numpy().copy()

class PassiveJointRecorder:
    def __init__(self, data, actuator_collection):
        assert data.has_joint_ordering, 'Identity ordering may alias public and backend state; not admitted.'
        self.data = data
        self.collection = actuator_collection
        self.order = copy(data.joint_ordering.user_to_backend).astype(np.int64)
        assert self.order.ndim == 1 and np.array_equal(np.sort(self.order),np.arange(len(self.order)))
        self.cache = [data._joint_pos, data._joint_vel]
        self.executor = actuator_collection._implicit_executor
        assert self.executor is not None
        self.public_pointers = [int(tensor(x.data).data_ptr()) for x in self.cache]
        self.kernel_pointers = [int(tensor(x).data_ptr()) for x in self.executor.kernel_inputs[3:5]]
        assert self.public_pointers == self.kernel_pointers
        self.backend_pointers = None
        self.records = []

    def capture(self, substep, time_s, command):
        # Access existing buffers only; no data.joint_pos/vel/body/IMU property.
        before = [copy(x.data) for x in self.cache]
        before_time = [float(x.timestamp) for x in self.cache]
        public_ptr = [int(tensor(x.data).data_ptr()) for x in self.cache]
        assert public_ptr == self.public_pointers
        backend = [self.data._root_view.get_dof_positions(), self.data._root_view.get_dof_velocities()]
        backend_ptr = [int(tensor(x).data_ptr()) for x in backend]
        for x in backend:
            lo = tensor(x).data_ptr(); hi = lo + tensor(x).numel()*tensor(x).element_size()
            for y in self.cache:
                a = tensor(y.data).data_ptr(); b = a + tensor(y.data).numel()*tensor(y.data).element_size()
                assert hi <= a or b <= lo, 'Backend query aliases public actuator state.'
        if self.backend_pointers is None:
            self.backend_pointers = backend_ptr
        assert backend_ptr == self.backend_pointers
        values = [copy(x) for x in backend]
        # Existing telemetry array is copied without reading any state properties.
        torque = copy(self.collection._applied_effort)
        after = [copy(x.data) for x in self.cache]
        after_time = [float(x.timestamp) for x in self.cache]
        assert before_time == after_time
        assert all(np.array_equal(a,b) for a,b in zip(before,after))
        assert [int(tensor(x).data_ptr()) for x in self.executor.kernel_inputs[3:5]] == self.kernel_pointers
        row = dict(substep=int(substep),t=float(time_s),q_backend=values[0],qd_backend=values[1],
            q=values[0][:,self.order].copy(),qd=values[1][:,self.order].copy(),
            public_q_before=before[0],public_q_after=after[0],
            public_qd_before=before[1],public_qd_after=after[1],
            cache_timestamp_before=np.array(before_time),cache_timestamp_after=np.array(after_time),
            command=command.detach().cpu().numpy().copy(),torque=torque)
        assert all(np.isfinite(x).all() for x in row.values())
        self.records.append(row)

    def metadata(self):
        return dict(nonidentity_joint_ordering=True,user_to_backend=self.order.tolist(),
            public_pointers=self.public_pointers,implicit_kernel_state_pointers=self.kernel_pointers,
            backend_pointers=self.backend_pointers,records=len(self.records),
            every_read_public_cache_bytes_and_timestamps_exact=True,
            every_backend_query_nonoverlapping_public_storage=True,control_use=False)
