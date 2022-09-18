#Basic run on cartpole - if this does something weird then you know something broke.
import gym
import sys
sys.path.append('./src')
import torch
device = torch.device("cuda")

env_id = 'CartPole-v1'
fbfile = 'testcartpole.dat'

env = gym.make(env_id)

env.reset()


from feedback import FeedbackManager

fb = FeedbackManager(fbfile, show_picker=True)

from rewardmodel import MlpRewardModel, ConvRewardModel
from rewardensemble import RewardEnsemble

def makemodelfun():
    def _f():
        return MlpRewardModel(env.observation_space.shape, device)
    return _f

#Ensemble just has 1 here just because i don't have enough comparisons on cartpole to do more.
rewardensemble = RewardEnsemble(1, makemodelfun())

from envwithrewardmodel import EnvWithRewardModel
def makeenvfun():
    def _f():
        #env_id specified higher in the notebook
        base_env = gym.make(env_id)
        return EnvWithRewardModel(base_env, rewardensemble)
    return _f



from stable_baselines3.common.vec_env import DummyVecEnv
vec_env = DummyVecEnv([makeenvfun() for i in range(4)])

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C



model = A2C("MlpPolicy", vec_env, verbose=1)

#Train reward model
def train_reward(n):
    for i in range(n):
        #batch = fb.randomBatch(20)
        #rewardmodel.train(batch)
        batches = fb.batchForEnsemble(20, rewardensemble.n)
        rewardensemble.train(batches)
        
        
#Train agent
def train_a2c(n):
    for i in range(n):
        model.learn(total_timesteps=1000, reset_num_timesteps=False)

        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print(f"{(i+1) * 1000}: mean {mean_reward}, std {std_reward}")
        

train_reward(100)
train_a2c(10)

from envplayer import EnvPlayer

player_env = gym.make(env_id)
player = EnvPlayer(player_env, model, rewardensemble)
