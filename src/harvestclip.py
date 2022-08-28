import random

def harvestClips(env, agent, n_timesteps=100, n_clips=2, clip_length=20):
    frames = []
    observations = []

    obs = env.reset()
    for i in range(n_timesteps):
        action, _ = agent.predict(obs)
        obs, _, done, _ = env.step(action)
        observations.append(obs)
        if done:
            env.reset()
        frames.append(env.render(mode="rgb_array"))

    clips = []
    obs_sets = []
    for i in range(n_clips):
        start = random.randrange(n_timesteps - clip_length)
        clips.append(frames[start:start+clip_length])
        obs_sets.append(observations[start:start+clip_length])

    return clips, obs_sets
