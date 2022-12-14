import gym
import numpy as np

class EnvWithRewardModel(gym.Env):
    def __init__(self, env, rewardmodel, episode_length=500, steps_per_reward=1):
        self.env = env
        self.rewardmodel = rewardmodel

        self.steps_per_reward=steps_per_reward

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        print(f"observation space: {self.observation_space.shape}")

        #Episode length is kept fixed regardless of the underlying environment.
        #Otherwise the agent could use episode length as a bootleg reward signal.
        self.episode_length = episode_length
        self.current_timestep = 0

        self.last_reward=0

    def reset(self):
        self.current_timestep = 0
        return self.env.reset()

    def step(self, action):
        self.current_timestep+=1
        my_done = (self.current_timestep == self.episode_length)

        state, envreward, done, info = self.env.step(action)
        if done:
            state = self.env.reset()

        state = np.array(state)
            
        reward = 0
        if self.current_timestep % self.steps_per_reward == 0:
            reward = self.rewardmodel.evaluate(state)

        #If env resets, deliver a higher-magnitude signal 
        #This either strongly rewards a good end to the env's episode or strongly punishes a bad end
        #(where "good" and "bad" are defined by the human feedback, not by the original environment.)
        #I think this is *close* to cheating, but still fair.
        #The RLHF paper did have something about penalties around episode reset
        if done:
            factor = 20 #arbitrary
            reward = self.last_reward * factor

        self.last_reward = reward

        return state, reward, my_done, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()
