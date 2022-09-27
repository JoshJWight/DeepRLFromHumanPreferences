from threading import Thread
import gym
import torch
import stable_baselines3.common.atari_wrappers as atari_wrappers
from rewardmodel import MlpRewardModel, ConvRewardModel
from rewardensemble import RewardEnsemble
from feedback import FeedbackManager
from envwithrewardmodel import EnvWithRewardModel
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
from harvestclip import harvestClips, harvestSynthetic, harvestWithEnsemble
from envplayer import EnvPlayer
import time
import statistics
from datetime import datetime

class TrainingBackend:
    def __init__(self, env_id, feedback_file, save_dir, conv=False, preprocess=False):
        self.env_id=env_id
        self.preprocess = preprocess
        self.conv = conv
        self.save_dir = save_dir

        #Signals from UI
        self.trainReward = False
        self.trainAgent = False
        self.harvestRandom = False
        self.harvestEnsemble = False
        self.shouldSave = False

        self.rewardIterations = 0
        self.agentIterations = 0

        self.lastAgentPrint = datetime.now()
        self.lastRewardPrint = datetime.now()


        self.device = torch.device("cuda")

        self.harvest_env = self.make_env()
        self.rewardensemble = RewardEnsemble(3, self.makemodelfun(self.harvest_env.observation_space.shape))

        self.feedback = FeedbackManager(feedback_file, show_picker=True)

        #Agent hyperparameters
        n_workers = 4
        #Actually, all the other a2c hyperparameters in the paper are the same as stable_baselines' defaults.
        #The paper does say they are "standard settings"
        #Except TODO the paper says lr decays over time but the default is "constant" lr_schedule

        vec_env = DummyVecEnv([self.makeenvfun() for i in range(n_workers)])

        self.agent_file = f"{self.save_dir}/agent/agent"
        try:
            self.agent = A2C.load(self.agent_file, vec_env)
            print("Loaded agent from file.")
        except:
            print("Could not load agent. Creating fresh agent.")
            if conv:
                self.agent = A2C("CnnPolicy", vec_env, verbose=1)
            else:
                self.agent = A2C("MlpPolicy", vec_env, verbose=1)


        #This is NOT THREAD SAFE
        #For now it's up to the user to make sure the env player is paused while any
        #other operations are underway.
        player_env = self.make_env()
        self.env_player = EnvPlayer(player_env, self.agent, self.rewardensemble) 

        self.thread = Thread(target=self.run)
        self.thread.start()

    def make_env(self):
        new_env = gym.make(self.env_id)
        if self.preprocess:
            new_env = gym.wrappers.AtariPreprocessing(new_env)
            new_env = gym.wrappers.FrameStack(new_env, 4)
            new_env = atari_wrappers.ClipRewardEnv(new_env)
        return new_env

    def makeenvfun(self):
        #steps_per_reward = 4
        steps_per_reward = 1

        def _f():
            base_env = self.make_env()
            return EnvWithRewardModel(base_env, self.rewardensemble, steps_per_reward = steps_per_reward)
        return _f

    def makemodelfun(self,input_shape):
        file_root = f"{self.save_dir}/reward/reward_model"
        if self.conv:
            def _f(model_num):
                return ConvRewardModel(input_shape, self.device, f"{file_root}_{model_num}.dat")
            return _f
        else:
            def _f(model_num):
                return MlpRewardModel(input_shape, self.device, f"{file_root}_{model_num}.dat")
            return _f

    def run(self):
        while True:
            REWARD_TRAIN_N = 100
            AGENT_TRAIN_N = 1000
            if self.trainReward:
                self.train_reward(REWARD_TRAIN_N)
            if self.trainAgent:
                self.train_agent(AGENT_TRAIN_N)
            if self.harvestRandom:
                self.harvest_random(1)
            if self.harvestEnsemble:
                self.harvest_ensemble(1)
            if self.shouldSave:
                self.save()
                self.shouldSave = False

            if not (self.trainReward or self.trainAgent or self.harvestRandom or self.harvestEnsemble):
                print("sleep")
                time.sleep(1)

    def save(self):
        self.feedback.save()
        self.rewardensemble.save()
        self.agent.save(self.agent_file)
        print("Saved!") 
                
    def train_reward(self, n):
        for i in range(n):
            self.rewardIterations += 1
            batches = self.feedback.batchForEnsemble(20, self.rewardensemble.n)
            self.rewardensemble.train(batches)

            if self.rewardIterations % 50 == 0:
                loss_averages = [statistics.mean(model.lossHistory) for model in self.rewardensemble.models]
                loss_strings = [f"{avg:.2f}" for avg in loss_averages]
                
                now = datetime.now()
                diff = now - self.lastRewardPrint
                self.lastRewardPrint = now
                print(f"{self.rewardIterations}: Reward nets' average losses: {loss_strings} ({diff})")

    def train_agent(self, n):
        self.agent.learn(total_timesteps=n, reset_num_timesteps=False)

        self.agentIterations += n
        if self.agentIterations % 1000 == 0:
            mean_reward, std_reward = evaluate_policy(self.agent, self.agent.get_env(), n_eval_episodes=10)

            now = datetime.now()
            diff = now - self.lastAgentPrint
            self.lastAgentPrint = now
            print(f"{self.agentIterations}: Agent reward: mean {mean_reward}, std {std_reward} ({diff})")

    def harvest_random(self, n):
        for i in range(n):
            clips, obs = harvestClips(self.harvest_env, self.agent, n_timesteps=300)
            self.feedback.queueClips(clips[0], clips[1], obs[0], obs[1])
            
    def harvest_ensemble(self,n):
        for i in range(n):
            clips, obs = harvestWithEnsemble(self.harvest_env, self.agent, self.rewardensemble, n_timesteps=300)
            self.feedback.queueClips(clips[0], clips[1], obs[0], obs[1])
            
    def add_synth_feedback(self,n):
        for i in range(n):
            print(i+1)
            self.feedback.addComparison(harvestSynthetic(self.harvest_env, self.agent, n_timesteps=300))
