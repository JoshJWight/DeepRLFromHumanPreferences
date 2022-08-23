import gym

class EnvWithRewardModel(gym.Env):
    def __init__(self, env, rewardmodel, episode_length=500):
        self.env = env
        self.rewardmodel = rewardmodel

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        #Episode length is kept fixed regardless of the underlying environment.
        #Otherwise the agent could use episode length as a bootleg reward signal.
        self.episode_length = episode_length
        self.current_timestep = 0

    def reset(self):
        self.current_timestep = 0
        return self.env.reset()

    def step(self, action):
        self.current_timestep+=1
        my_done = (self.current_timestep == self.episode_length)

        state, _, done, info = self.env.step(action)
        reward = 0
        if done:
            state = self.env.reset()
        else:
            reward = self.rewardmodel.evaluate(state)
        
        return state, reward, my_done, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()
