import pickle
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim



#RLHF paper, appendix A.2

#This expects the input to have the channels as the first dimension
#Which is true with the frame stack wrapper but not with the base atari environments.
class ConvRewardNet(nn.Module):
    def __init__(self, input_shape):
        super(ConvRewardNet, self).__init__()

        #With the frame stack environment the input channels are the greyscale values for the pixel from the different frames
        n_input_channels = input_shape[0]

        #print(f"n_input_channels: {n_input_channels}")
        self.conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=7, stride=3),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Conv2d(16, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.01)
        )
            #Flatten the 3-dimensional conv output into 1-dimensional linear input
            #But leave any batch dimensions intact 
        #self.flatten = nn.Flatten(-3)
        conv_out_size = self._get_conv_out(input_shape)
        
        self.value = nn.Sequential(
            #Adjusts to whatever the input image size happens to be.
            nn.Linear(conv_out_size, 64),
            #The paper doesn't explicitly say there's a nonlinearity here but I think it makes sense.
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            #TODO this is not what the paper says.
            #Paper says they "normalize it to have a standard deviation of 0.05"
            nn.Sigmoid()
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        #print(x)
        fx = x.float() / 256

        #Gym outputs a tensor with the color channels as the third dimension
        #But torch expects channels to be the first dimension.
        #Doing this one flip does mean that the x and y dimensions are swapped, 
        #but that shouldn't matter for net performance as long as it's consistent
        #fx = x.transpose(-1, -3)
        #print(fx)
        #print(f"input shape: {x.shape}")

        fx = self.conv(fx)
        #print(fx)
        #fx = self.flatten(fx)
        #size()[0] is the batch dimension
        fx = fx.view(fx.size()[0], -1)
        #print(fx)
        fx = self.value(fx)
        #print(fx)

        return fx

#RLHF paper, appendix A.1
class MlpRewardNet(nn.Module):
    def __init__(self, n_inputs):
        super(MlpRewardNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class RewardModel:
    def __init__(self, net, device, file, lr):
        self.modelFile = file

        self.device = device
        self.net = net

        #epsilon = 1e-3
        epsilon = 1e-5
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=epsilon)

        self.lossHistory = collections.deque(maxlen=20)
    
        try:
            self.net.load_state_dict(torch.load(self.modelFile))
            self.net.eval()
            print("Loaded reward model")
        except:
            print("Could not load reward model")

    def save(self):
        torch.save(self.net.state_dict(), self.modelFile)
    
    def evaluate(self, state):
        #BatchNorm in the conv net requires input to always be in batch form.
        state = np.array([state])
        #TODO the paper recommends regularization since the scale of the reward model is arbitrary.
        stateTensor = torch.FloatTensor(state).to(self.device)
        n = self.net(stateTensor).detach().item()#this will only work if we're evaluating a single state
        return (n - 0.5) * 2.0

    def evaluateMany(self, states):
        stateTensor = torch.FloatTensor(states).to(self.device)
        output = self.net(stateTensor).detach().sub(0.5).mul(2.0).cpu().numpy()
        return output
        

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
            
            evals1 = self.net(clip1_v)
            evals2 = self.net(clip2_v)            

            expsum1 = torch.exp(evals1.sum())
            expsum2 = torch.exp(evals2.sum())

            phat1 = expsum1 / (expsum1 + expsum2)
            phat2 = expsum2 / (expsum2 + expsum1)

            loss = -1.0 * ((p1 * torch.log(phat1)) + (p2 * torch.log(phat2)))
            losses.append(loss)

            #Second loss component: penalty for both segments being all really close to zero or one
            #This was not in the paper; I'm just hacking around a problem I'm having.
            #Edit: my real problem was that my learn rate was too high and this is probably useless. Still want it in version control though.
            '''
            mean1 = torch.mean(evals1)
            mean2 = torch.mean(evals2)
            #This is equivalent to taking the mean of the means and multiplying by 2
            mean = torch.add(mean1, mean2)
            adjmean = mean.sub(1)
            loss2 = adjmean.pow(4) 
            losses.append(loss2)
            '''
            #print(f"evals1: {evals1}, evals2: {evals2}")
            #print(f"expsums: {expsum1}, {expsum2} phats: {phat1}, {phat2}, p: {p1}, {p2} loss: {loss}")
            #print(f"means: {mean1} + {mean2} = {mean}, adjusted to {adjmean}, final loss2 {loss2}")
            if torch.isnan(loss).any():
                print("NaN Alert!")
                #print(f"evals1: {evals1}, evals2: {evals2}")
                print(f"expsums: {expsum1}, {expsum2} phats: {phat1}, {phat2} loss: {loss}")
                #for notebook debugging
                self.naneval = evals1

            #TODO track stuff

        totalLoss = sum(losses)
        totalLoss.backward()
        self.optimizer.step()

        #print(f"Reward model loss: {totalLoss}")
        self.lossHistory.append(totalLoss.item())
             
def MlpRewardModel(input_shape, device, file, lr=0.01):
    net = MlpRewardNet(input_shape[0]).to(device)
    return RewardModel(net, device, file, lr)


#TODO yes we established that this lr is better than 0.01, but is something else better still?
def ConvRewardModel(input_shape, device, file, lr=0.001):
    net = ConvRewardNet(input_shape).to(device)
    return RewardModel(net, device, file, lr)

