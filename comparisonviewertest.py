import gym
import sys
sys.path.append('./src')

from feedback import FeedbackManager

fb = FeedbackManager("testcartpole.dat")

fb.viewComparisons()
