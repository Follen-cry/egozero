from gym.envs.registration import register

# Register an Aloha environment with the same interface as AlohaEnv
register(
    id="Aloha-v1",
    entry_point="aloha_env.envs:AlohaEnv",
    max_episode_steps=400,
)

