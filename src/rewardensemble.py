
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
        evaluations = []
        for obs_set in obs_sets:
            #TODO this doesn't work because rewardmodel.evaluate() only takes one state.
            evaluations.append([model.evaluateMany(obs_set) for model in self.models])

        sums = []
        disagreements = []
        for i in range(len(evaluations)):
            averages.append(sum(evaluations[i]))
            max_disagreement = 0
            for j1 in range(self.n):
                for j2 in range(i+1, self.n):
                    disagreement = abs(evaluations[i][j1] - evaluations[i][j2])
                    if disagreement > max_disagreement:
                        max_disagreement = disagreement
            disagreements.append(max_disagreement)

        #Pick the "experimental" clip
        #This is the one with the highest disagreement.
        exp_idx = 0
        for i in range(1, len(evaluations)):
            if disagreements[i] > disagreements[exp_idx]:
                exp_idx = i


        #Pick the "control" clip
        #Random selection from the other clips weighted by
        # - Similar average score to experimental clip
        # - Low disagreement
        compatibilities = []
        for i in range(len(evaluations)):
            a = 1 / (abs(sums[i] - sums[exp_idx]) + 0.1)
            #b will always be 0 at exp_idx, so no weight on picking the experimental clip again.
            b = disagreements[exp_idx] - disagreements[i]
            compatibilities.append(a * b)

        ctl_idx = random.choices(range(len(evaluations)), weights=compatibilities, k=1)

        return (exp_idx, ctl_idx)
