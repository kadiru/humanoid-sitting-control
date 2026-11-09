"""Discoverable Isaac Lab task registrations."""
import gymnasium as gym

gym.register(
    id="Humanoid-Sit-Fixed-G1",
    entry_point="humanoid_sitting.tasks.env:SittingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "humanoid_sitting.tasks.config:SittingEnvCfg",
        "rsl_rl_cfg_entry_point": "humanoid_sitting.tasks.config:SittingPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

for stage, config in [(0, 'SeatedStabilityEnvCfg'), (1, 'NearSeatEnvCfg'),
                      (2, 'PartialDescentEnvCfg'), (3, 'StandingTransitionEnvCfg')]:
    gym.register(
        id=f"Humanoid-Sit-Curriculum-{stage}-G1",
        entry_point="humanoid_sitting.tasks.env:SittingEnv", disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"humanoid_sitting.tasks.curriculum:{config}",
                "rsl_rl_cfg_entry_point": "humanoid_sitting.tasks.curriculum:CurriculumPPORunnerCfg",
                "default_agent": "rsl_rl"},
    )

gym.register(
    id='Humanoid-Sit-Full-Transition-G1',
    entry_point='humanoid_sitting.tasks.env:SittingEnv',disable_env_checker=True,
    kwargs={'env_cfg_entry_point':'humanoid_sitting.tasks.full_transition:FullTransitionEnvCfg',
            'rsl_rl_cfg_entry_point':'humanoid_sitting.tasks.full_transition:FullTransitionPPORunnerCfg',
            'default_agent':'rsl_rl'},
)

gym.register(
    id='Humanoid-Sit-Rigid-Furniture-G1',
    entry_point='humanoid_sitting.tasks.env:SittingEnv',disable_env_checker=True,
    kwargs={'env_cfg_entry_point':'humanoid_sitting.tasks.furniture:FurnitureEnvCfg',
            'rsl_rl_cfg_entry_point':'humanoid_sitting.tasks.full_transition:FullTransitionPPORunnerCfg',
            'default_agent':'rsl_rl'},
)

# The existing fixed-seat and furniture task configurations remain intact.
gym.register(
    id='Humanoid-Sit-Geometry-Conditioned-G1',
    entry_point='humanoid_sitting.tasks.env:SittingEnv',disable_env_checker=True,
    kwargs={'env_cfg_entry_point':'humanoid_sitting.tasks.geometry_curriculum:GeometryV12TrainingEnvCfg',
            'rsl_rl_cfg_entry_point':'humanoid_sitting.tasks.full_transition:FullTransitionPPORunnerCfg',
            'default_agent':'rsl_rl'},
)
