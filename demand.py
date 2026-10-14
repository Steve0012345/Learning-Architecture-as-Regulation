import numpy as np


def choice_utilities(prices, veh_counts, total_vehs, beta_price=1.0, beta_avail=0.5, brand_prefs=None):

    num_plat = len(prices)
    if brand_prefs is None:
        brand_prefs = [0.0] * num_plat
    # fractional availability per zone per platform 
    avails = [veh_counts[k] / (total_vehs[k] + 1e-4) for k in range(num_plat)]
    return np.array([brand_prefs[k] - beta_price * prices[k] + beta_avail * avails[k] for k in range(num_plat)])


def choice_probabilities(V):
    # numerically stable softmax over platforms, sums to 1 per origin zone
    exp_V = np.exp(V - V.max(axis=0))
    return exp_V / exp_V.sum(axis=0)


def split_demand(dd_block, prices, veh_counts, total_vehs, beta_price=1.0, beta_avail=0.5, brand_prefs=None):
    """Assign each rider in dd_block to one of N platforms using a logit choice model
    """
    num_plat = len(prices)

    single = dd_block.ndim == 2
    if single:
        dd_block = dd_block[None, :, :]

    freq, num_zone, _ = dd_block.shape


    V = choice_utilities(prices, veh_counts, total_vehs, beta_price, beta_avail, brand_prefs)
    probs = choice_probabilities(V)  # (num_plat, num_zone)

    # Allocate riders using sequential binomial
    # For each origin zone i, riders independently choose a platform with prob probs[:, i].
    # The stick-breaking approach is vectorized over destinations within each origin zone:
    # assign to platform 0 first, then from remaining assign to platform 1, etc.
    demands = [np.zeros_like(dd_block, dtype=int) for _ in range(num_plat)]
    for step in range(freq):
        remaining = dd_block[step].copy().astype(int)
        for k in range(num_plat - 1):
            # conditional probability of choosing platform k given not yet assigned
            tail_prob = probs[k:, :].sum(axis=0) + 1e-12  # (num_zone,)
            cond_prob = probs[k, :] / tail_prob             # (num_zone,), per origin zone
            for i in range(num_zone):
                if np.any(remaining[i, :] > 0):
                    demands[k][step, i, :] = np.random.binomial(remaining[i, :], cond_prob[i])
            remaining -= demands[k][step]
        # last platform gets whoever wasn't allocated
        demands[num_plat - 1][step] = remaining

    if single:
        return [demands[k][0] for k in range(num_plat)]
    return demands
