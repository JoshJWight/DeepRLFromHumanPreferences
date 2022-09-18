import random
import numpy as np

class RewardEnsemble:
    def __init__(self, n_models, model_fn):
        self.n = n_models
        self.models = [model_fn() for i in range(n_models)]

    def evaluate(self, state):
        return sum([model.evaluate(state) for model in self.models]) / self.n

    def train(self, sample_sets):
        for model, samples in zip(self.models, sample_sets):
            model.train(samples) 

    def pickComparison(self, obs_sets):
        print("Reward ensemble picking comparison")

        #2d array: for each clip, for each model in the ensemble
        evaluations = []
        for obs_set in obs_sets:
            #Right at the start, sum the evaluations of all frames in a clip together.
            #From here on, we will be comparing *clips*, not *frames*
            evaluations.append([np.sum(model.evaluateMany(obs_set)) for model in self.models])

        print(f"Evaluations: {evaluations}")

        #1d array: the sum of all models' evaluations for each clip
        sums = []
        #1d array: the greatest disagreement between two models in the ensemble for each clip
        disagreements = []
        for i in range(len(evaluations)):
            sums.append(sum(evaluations[i]))
            max_disagreement = 0
            for j1 in range(self.n):
                for j2 in range(j1+1, self.n):
                    disagreement = abs(evaluations[i][j1] - evaluations[i][j2])
                    if disagreement > max_disagreement:
                        max_disagreement = disagreement
            disagreements.append(max_disagreement)

        print(f"Sums: {sums}")
        print(f"Disagreements: {disagreements}")

        #Pick the "experimental" clip
        #This is the one with the highest disagreement.
        exp_idx = 0
        for i in range(1, len(evaluations)):
            if disagreements[i] > disagreements[exp_idx]:
                exp_idx = i
        
        print(f"Experimental index: {exp_idx}")

        #Pick the "control" clip
        #Random selection from the other clips weighted by
        # - Similar average score to experimental clip
        # - Low disagreement
        compatibilities = []
        for i in range(len(evaluations)):
            a = 1 / (abs(sums[i] - sums[exp_idx]) + 0.1)
            #b will always be 0 at exp_idx, so no weight on picking the experimental clip again.
            b = disagreements[exp_idx] - disagreements[i]
            c = a*b
            #Raise them to a power to make the best ones more likely to be picked and the worst ones less
            compatibilities.append(c ** 2)

        print(f"Compatibilities: {compatibilities}")

        ctl_idx = random.choices(range(len(evaluations)), weights=compatibilities, k=1)[0]

        print(f"Control index: {ctl_idx}")

        print(f"Experimental: evaluations {evaluations[exp_idx]}, sum {sums[exp_idx]}, disagreement {disagreements[exp_idx]}")
        print(f"Control: evaluations {evaluations[ctl_idx]}, sum {sums[ctl_idx]}, disagreement {disagreements[ctl_idx]}")

        return exp_idx, ctl_idx
