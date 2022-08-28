import pickle
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

#Note that this design doesn't care about the action the agent took, only the state it reached.
#This means it can't really give an opinion on actions that end the episode.
class ConvRewardNet(nn.Module):
    def __init__(self, input_shape):
        super(ConvRewardNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def _get_conv_out(self, shape):
        #TODO i think this flattens even multiple batches together. gotta fix this
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.value(conv_out)

class MlpRewardNet(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super(MlpRewardNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class RewardModel:
    def __init__(self, net, device, lr=0.01, load=False):
        self.modelFile = "rewardmodel.dat"

        self.device = device
        self.net = net

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-3)
    
        print(f"Load: {load}")
        if load:
            self.net.load_state_dict(torch.load(self.modelFile))
            self.net.eval()
            print("Loaded reward model")

    def save(self):
        torch.save(self.net.state_dict(), self.modelFile)
    
    def evaluate(self, state):
        #TODO the paper recommends regularization since the scale of the reward model is arbitrary.
        stateTensor = torch.FloatTensor(state).to(self.device)
        n = self.net(stateTensor).detach().item()#this will only work if we're evaluating a single state
        return (n - 0.5) * 2.0

    def train(self, samples):
        losses = []
        for comparison in samples:
            clip1 = comparison["observations"][0]
            clip2 = comparison["observations"][1]
            p1    = comparison["values"][0]
            p2    = comparison["values"][1]
            
            #Might be able to do this in a single pass with a higher-dimensional tensor

            #assume clips are arrays of states
            clip1_v = torch.FloatTensor(np.array(clip1, copy=False)).to(self.device)
            clip2_v = torch.FloatTensor(np.array(clip2, copy=False)).to(self.device)

            self.optimizer.zero_grad()

            expsum1 = torch.exp(self.net(clip1_v).sum())
            expsum2 = torch.exp(self.net(clip2_v).sum())

            phat1 = expsum1 / (expsum1 + expsum2)
            phat2 = expsum2 / (expsum2 + expsum1)

            loss = -1.0 * ((p1 * torch.log(phat1)) + (p2 * torch.log(phat2)))
            losses.append(loss)
            
            if torch.isnan(loss).any():
                print("NaN Alert!")
                print(f"expsums: {expsum1}, {expsum2} phats: {phat1}, {phat2} loss: {loss}")

            #TODO track stuff

        totalLoss = sum(losses)
        totalLoss.backward()
        self.optimizer.step()

        print(f"Reward model loss: {totalLoss}")
             
def MlpRewardModel(input_shape, device, load=False, n_hidden=10):
    net = MlpRewardNet(input_shape[0], n_hidden=n_hidden).to(device)
    return RewardModel(net, device, load)


def ConvRewardModel(input_shape, device, load=False):
    net = ConvRewardNet(input_shape).to(device)
    return RewardModel(net, device, load)

