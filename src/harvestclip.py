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

    startrange = n_timesteps - clip_length
    rangeperclip = int(startrange / n_clips)

    for i in range(n_clips):
        start = random.randrange(rangeperclip) + (i * rangeperclip)
        clips.append(frames[start:start+clip_length])
        obs_sets.append(observations[start:start+clip_length])

    return clips, obs_sets

def harvestWithEnsemble(env, agent, reward_ensemble, n_timesteps=100, clip_length=20):
    obs = env.reset()

    clips = []
    obs_sets = []

    for i in range(int(n_timesteps/clip_length)):
        clip = []
        obs_set = []
        for j in range(clip_length):
            action, _ = agent.predict(obs)
            obs, _, done, _ = env.step(action)
            obs_set.append(obs)
            if done:
                env.reset()
            clip.append(env.render(mode="rgb_array"))
        clips.append(clip)
        obs_sets.append(obs_set)
    
    idx1, idx2 = reward_ensemble.pickComparison(obs_sets)
    print(f"idx1: {idx1}, idx2: {idx2}")
    return [clips[idx1], clips[idx2]], [obs_sets[idx1], obs_sets[idx2]]
            
    

def harvestSynthetic(env, agent, n_timesteps=100, clip_length=20, render=False):
    frames = []
    observations = []
    rewards = []

    obs = env.reset()
    for i in range(n_timesteps):
        action, _ = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        if done:
            env.reset()
        if render:
            frames.append(env.render(mode="rgb_array"))

    clips = []
    obs_sets = []
    rew_sets = []

    startrange = n_timesteps - clip_length
    rangeperclip = int(startrange / 2)

    for i in range(2):
        start = random.randrange(rangeperclip) + (i * rangeperclip)
        clips.append(frames[start:start+clip_length])
        obs_sets.append(observations[start:start+clip_length])
        rew_sets.append(rewards[start:start+clip_length])

    rew_sums = [sum(x) for x in rew_sets]

    values = [0.5, 0.5]
    if rew_sums[0] > rew_sums[1]:
        values = [1, 0]
    elif rew_sums[0] < rew_sums[1]:
        values = [0, 1]

    result = {
        "clips": clips,
        "observations": obs_sets,
        "values": values 
    }
    return result
