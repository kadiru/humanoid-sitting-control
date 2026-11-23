import sys
from latched_seat_run import P,launch,inputs
if __name__=='__main__':
    new=['supported_touchdown_motion.py','aimed_replant_motion.py','aimed_replant_paths.json','aimed_replant_paths.npz','aimed_step_primitive.npz','three_support_anchor.json','three_support_plan.py','three_support_motion.py','quiet_stance_motion.py','quiet_stance_run.py','self_approach_plan.py','self_approach_paths.npz','self_approach_specs.json','bilateral_rim_plan.py','bilateral_rim_paths.npz','bilateral_rim_targets.json','short_perch_pose.py','short_perch_pose.json','timed_perch_sweep.py','timed_thigh_sweep.json','time_perch_commands.py','perch_command_sweep.npz','perch_command_timing.json','impact_order_checked_motion.py','impact_order_checked_run.py']
    new+=['planted_edge_search.py','planted_edge_fixed_search.py','contact_corridor_fit.py','loaded_corridor_fit.py','contact_corridor_path.py','planted_corridor_motion.py','planted_corridor_run.py','planted_corridor_routes.json','corridor_pressure_motion.py','corridor_pressure_run.py','stance_frame_perch.py','stance_frame_perch.npz','sole_centered_mount.py']
    new+=['sole_moment_routes.json','sole_moment_motion.py','sole_moment_run.py','corridor_trace_correction.py','corridor_trace_correction.npz','hover_edge_graph.py','hover_edge_path_12.json','hover_edge_path_12.npz','flat_edge_clearance_12.json','loaded_corridor_12.json']
    new+=['measured_corner_motion.py','measured_corner_run.py','measured_corner_path.py','measured_corner_path_30.npz','measured_corner_routes.json','first_touch_trim.py','first_touch_trim_30.json','first_touch_trim_30.npz','measured_corner_timing.json']
    new+=['supported_corner_transfer.py','supported_corner_transfer.npz','supported_corner_run.py']
    new+=['loaded_corner_motion.py','loaded_corner_fit.py','loaded_corner_fit.npz','loaded_corner_run.py']
    new+=['corner_traction_fit.py','corner_traction_fit.npz','corner_traction_motion.py','corner_traction_run.py']
    for i in [6,17,21,30]:new.extend([f'contact_corridor_path_{i}.npz',f'contact_corridor_path_{i}.json',f'loaded_corridor_{i}.json'])
    for arg in sys.argv[1:]:
        ids=[30] if arg=='monarch' else [0,6,12,17,18,21,26,30] if arg=='pilot' else list(range(int(arg),int(arg)+16));launch('corner_traction_'+arg,ids,worker='corner_traction_worker.py',more=[P/n for n in inputs+new])

