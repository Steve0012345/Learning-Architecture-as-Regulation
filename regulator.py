import numpy as np
import torch
from torch.optim import Adam
from metrics import OUTCOME_NAMES, OUTCOME_SIGNS


class Regulator:

    def __init__(self, lambdas, guidance_dim=1, lr=1e-3, hidden=32, guidance_std=0.3, batch_intervals=32, eps=1e-6):
        self.lambdas = np.array([lambdas[name] for name in OUTCOME_NAMES])
        self.guidance_dim = guidance_dim
        self.guidance_std = guidance_std
        self.batch_intervals = batch_intervals
        self.eps = eps
        num_outcomes = len(OUTCOME_NAMES)
        self.policy = torch.nn.Sequential(torch.nn.Linear(num_outcomes, hidden), torch.nn.Tanh(),
                                          torch.nn.Linear(hidden, guidance_dim))
        self.optim = Adam(self.policy.parameters(), lr=lr)

        # Welford running mean / variance of the raw outcome vector
        self.outcome_count = 0
        self.outcome_mean = np.zeros(num_outcomes)
        self.outcome_m2 = np.zeros(num_outcomes)

        self.last_obs = None
        self.last_sample = None
        self.transitions = []
        self.history = []

    def standardize(self, outcome, update):
        raw = np.array([outcome[name] for name in OUTCOME_NAMES], dtype=float)
        raw = np.where(np.isnan(raw), self.outcome_mean, raw)
        if update:
            self.outcome_count += 1
            delta = raw - self.outcome_mean
            self.outcome_mean += delta / self.outcome_count
            self.outcome_m2 += delta * (raw - self.outcome_mean)
        std = np.sqrt(self.outcome_m2 / max(self.outcome_count, 1))
        return (raw - self.outcome_mean) / (std + self.eps)

    def observe(self, outcome, explore=True):
        """Close the interval that just ended and return the guidance for the next one."""
        if outcome is None:
            obs = np.zeros(len(OUTCOME_NAMES))
        else:
            obs = self.standardize(outcome, update=explore)
            score = float(obs @ (OUTCOME_SIGNS * self.lambdas))
            self.history.append(dict(outcome, score=score))
            if explore and self.last_obs is not None:
                self.transitions.append((self.last_obs, self.last_sample, score))
                if len(self.transitions) >= self.batch_intervals:
                    self.update()

        with torch.no_grad():
            mean = torch.tanh(self.policy(torch.from_numpy(obs).float()))
            sample = mean + torch.randn(self.guidance_dim) * self.guidance_std if explore else mean
        self.last_obs = obs
        self.last_sample = sample
        return sample.clamp(-1, 1).numpy().astype(float)

    def update(self):
        obs = torch.tensor(np.array([tr[0] for tr in self.transitions]), dtype=torch.float32)
        samples = torch.stack([tr[1] for tr in self.transitions])
        scores = torch.tensor([tr[2] for tr in self.transitions], dtype=torch.float32)
        # batch-mean baseline keeps the REINFORCE gradient centred
        advantage = scores - scores.mean()
        # observe() is called from inside the simulator's no_grad block
        with torch.enable_grad():
            dist = torch.distributions.Normal(torch.tanh(self.policy(obs)), self.guidance_std)
            loss = -(advantage * dist.log_prob(samples).sum(dim=1)).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        self.transitions = []

    def save(self, path):
        torch.save({'policy': self.policy.state_dict(), 'count': self.outcome_count,
                    'mean': self.outcome_mean, 'm2': self.outcome_m2}, path)

    def load(self, path):
        saved = torch.load(path, weights_only=False)
        self.policy.load_state_dict(saved['policy'])
        self.outcome_count, self.outcome_mean, self.outcome_m2 = saved['count'], saved['mean'], saved['m2']
