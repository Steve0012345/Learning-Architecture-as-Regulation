import numpy as np
from demand import choice_utilities, choice_probabilities

OUTCOME_NAMES = ['rs', 'served', 'tv', 'sd', 'gini_rs']
OUTCOME_SIGNS = np.array([1.0, 1.0, -1.0, -1.0, -1.0])


def log_sum_exp(V):
    V_max = V.max(axis=0)
    return V_max + np.log(np.exp(V - V_max).sum(axis=0))


def market_snapshot(prices, veh_counts, total_vehs, pool_zone, beta_price=1.0, beta_avail=0.5, brand_prefs=None,
                    max_price=3.0):
    """Rider-surplus index RS_i and expected price p_bar_i for one pricing interval

    pool_zone is the arriving rider pool per origin zone, D0_i = sum_j D0_ij, before it is
    split. p_bar_i = sum_k sigma_ki * p_ki. Availability must be pre-decision, the same idle
    counts the logit split sees.

    The log-sum is identified only up to an additive constant (Small and Rosen 1981; Train 2009),
    so the index is anchored at a reference market where every platform charges max_price and has
    no idle vehicles: RS_i = D0_i / beta_p * [log sum_k exp(V_ki) - log sum_k exp(V_ref_k)].
    The reference only fixes the zero of the measurement, and it doesn't enters pricing or rewards.
    Since p <= max_price and availability >= 0, the index is non-negative and the Gini is defined.
    """
    num_plat = len(prices)
    V = choice_utilities(prices, veh_counts, total_vehs, beta_price, beta_avail, brand_prefs)
    V_ref = choice_utilities([np.full_like(prices[k], max_price) for k in range(num_plat)],
                             [np.zeros_like(veh_counts[k]) for k in range(num_plat)],
                             total_vehs, beta_price, beta_avail, brand_prefs)
    rs_zone = pool_zone * (log_sum_exp(V) - log_sum_exp(V_ref)) / beta_price
    pbar_zone = np.sum(choice_probabilities(V) * np.stack(prices), axis=0)
    return rs_zone, pbar_zone


def zone_gini(values):

    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    mean = values.mean() if n > 0 else 0.0
    if mean <= 0:
        return np.nan
    # sum_ij |x_i - x_j| = 2 * sum_i (2i - n - 1) x_(i) for sorted x, avoids the N^2 matrix
    rank_weight = 2 * np.arange(1, n + 1) - n - 1
    return float(np.dot(rank_weight, values) / (n * n * mean))


class MarketRecord:

    def __init__(self):
        self.rs_zone = []
        self.pbar_zone = []
        self.served = 0

    def record(self, rs_zone, pbar_zone):
        self.rs_zone.append(rs_zone)
        self.pbar_zone.append(pbar_zone)

    def summary(self):
        if len(self.rs_zone) == 0:
            return None
        rs_hist = np.array(self.rs_zone)
        pbar_hist = np.array(self.pbar_zone)
        ginis = np.array([zone_gini(rs) for rs in rs_hist])
        return {
            'rs': float(rs_hist.sum()),
            'served': float(self.served),
            # TV: within-zone std over time, averaged over zones
            'tv': float(np.mean(np.std(pbar_hist, axis=0))),
            # SD: cross-zone std at each time, averaged over time
            'sd': float(np.mean(np.std(pbar_hist, axis=1))),
            # periods with an undefined Gini (non-positive mean index) are left out of the average
            'gini_rs': float(np.nanmean(ginis)) if np.any(~np.isnan(ginis)) else np.nan,
        }


def consumer_surplus(mu0_zone, price_zone, share_zone, maxp=3.0):
    """Expected consumer surplus per rider, summed over zones."""
    price_zone = np.maximum(price_zone, 1e-6)
    # zones where price >= maxp have zero CS
    cs_per_zone = np.where(price_zone < maxp,
                           mu0_zone * np.log(maxp / price_zone),
                           0.0)
    return float(np.sum(share_zone * cs_per_zone))


def price_volatility(price_history):
    """Standard deviation of price multipliers across all zones and time steps.
    """
    return float(np.std(price_history))


def gini(values):
    """Gini coefficient of a non-negative array.
    """
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.dot(idx, values) / (n * np.sum(values))) - (n + 1) / n)
