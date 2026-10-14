import numpy as np
import pandas as pd
from metrics import OUTCOME_NAMES, OUTCOME_SIGNS


def clone_state(module):
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


class FederatedServer:
    """Regulator-weighted actor aggregation.

    Each round, every platform submits its locally trained actor as a candidate. The regulator
    scores each candidate from a validation rollout against the round-start competitor actors,
    turns the scores into softmax weights and averages the actors.
    """

    def __init__(self, lambdas, tau=0.5, share_beta=1.0, eps=1e-6):
        self.lambdas = np.array([lambdas[name] for name in OUTCOME_NAMES])
        self.tau = tau
        self.share_beta = share_beta
        self.eps = eps
        self.reference_actors = None  
        self.history = []

    def snapshot(self, controllers):
        self.reference_actors = [clone_state(ctrl.pricer.actor) for ctrl in controllers]

    def score(self, outcomes):
        """q_k = sum_j sign_j * lambda_j * standardized m_k^(j), standardized across the round's candidates."""
        outcome_table = np.array([[m[name] for name in OUTCOME_NAMES] for m in outcomes], dtype=float)
        # population std across candidates
        standardized = (outcome_table - np.nanmean(outcome_table, axis=0)) / (np.nanstd(outcome_table, axis=0) + self.eps)
        standardized = np.nan_to_num(standardized, nan=0.0)
        return standardized @ (OUTCOME_SIGNS * self.lambdas)

    def aggregate(self, controllers, outcomes):
        scores = self.score(outcomes)
        exp_q = np.exp((scores - scores.max()) / self.tau)
        weights = exp_q / exp_q.sum()

        # The target actor is aggregated with the same weights
        for net_name in ['actor', 'actor_target']:
            candidates = [clone_state(getattr(ctrl.pricer, net_name)) for ctrl in controllers]
            global_actor = {}
            for name, tensor in candidates[0].items():
                if tensor.is_floating_point():
                    global_actor[name] = sum(float(weights[k]) * candidates[k][name] for k in range(len(candidates)))
                else:
                    global_actor[name] = tensor
            for k, ctrl in enumerate(controllers):
                mixed = {}
                for name, tensor in candidates[k].items():
                    if tensor.is_floating_point():
                        mixed[name] = (1 - self.share_beta) * tensor + self.share_beta * global_actor[name]
                    else:
                        mixed[name] = tensor
                getattr(ctrl.pricer, net_name).load_state_dict(mixed)

        round_log = {'round': len(self.history)}
        for k, m in enumerate(outcomes):
            for name in OUTCOME_NAMES:
                round_log[name + '_' + str(k)] = m[name]
            round_log['score_' + str(k)] = scores[k]
            round_log['weight_' + str(k)] = weights[k]
        self.history.append(round_log)

        # the adopted actors become the reference competitors for the next round
        self.snapshot(controllers)
        return weights

    def save_history(self, path):
        # audit record of every round
        pd.DataFrame(self.history).to_csv(path, index=None)
