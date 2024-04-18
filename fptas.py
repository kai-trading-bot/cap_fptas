import os
import matplotlib.pyplot as plt
import numpy as np
import pprint
import pandas as pd
import seaborn as sns
import warnings
from itertools import combinations

HOME = os.environ['HOME'] + '/Documents/cap_fptas/'

plt.style.use('seaborn-whitegrid')
warnings.filterwarnings("ignore")


def get_vstar(prob: pd.DataFrame, T: float) -> float:

    """
    Fetch school with highest utility that satisfies risk
    :param prob:
    :param T:
    :return:
    """
    select = prob[prob.PROBABILITY >= T].sort_values('UTILITY', ascending=False).reset_index(drop=True)
    if len(select) > 1:
        select = select.iloc[0]
        return select.loc['UTILITY'] * select.loc['PROBABILITY']
    return 1


def get_delta(v_star: float, eps: float, K: int) -> float:
    """
    d = (epsilon x V*) / K
    :param v_star:
    :param eps:
    :param K:
    :return:
    """
    return (v_star * eps) / K


def rounding(eu: float, delta: float) -> float:
    """
    Multiples of delta
    RU = (U / d) x d
    :param eu:
    :param delta:
    :return:
    """
    return np.round(eu / delta) * delta


def fptas(prob: pd.DataFrame,
          n: int,
          m: int,
          T: float,
          K: int,
          eps: float,
          verbose: bool = False) -> pd.DataFrame:
    """
    Fully polynomial time approximation scheme
    :param prob:
    :param n:
    :param m:
    :param T:
    :param K:
    :param eps:
    :param verbose:
    :return:
    """
    # Initialize hashmap
    F, G = {}, {}
    F[0, 0, 0] = 0
    G[0, 0, 0] = 0

    # Get v* and delta from settings
    v_star = get_vstar(prob=prob, T=T, K=K, eps=eps)
    delta = get_delta(v_star=v_star, eps=eps, K=K)

    for n in range(1, K + 1):
        for m in range(1, n + 1):
            if m == 1:
                sub = prob.head(n)
                for _, _, p, _, eu in sub.values:
                    rounded_utility = rounding(eu=eu, delta=delta)

                    # F[n, 1, u_i] = p_i
                    # F[n, 1, u > u_max] = 0
                    F[n, m, rounded_utility] = p
                    F[n, m, np.ceil(rounded_utility)] = 0
                    G[n, m, eu] = p
            else:
                sub = prob.head(n)
                indices = list(sub.index)
                combos = list(combinations(indices, m))

                # Check all possible combinations
                for combo in combos:
                    combo = list(combo)
                    tray = sub.loc[combo]
                    rest = tray.iloc[:-1]
                    last = tray.iloc[[-1]]
                    _, _, lp, _, leu = last.values[0]

                    # p_i + (q_i * (prod[k=1, k=i-1](1 - q_k)))
                    # u_i + (q_i * sum[k=1, k=i-1]u_k)
                    total_probability = lp + ((1 - lp) * (1 - (1 - rest.PROBABILITY).prod()))
                    utility = (rest.EU.sum() * (1 - lp)) + leu
                    rounded_utility = rounding(utility, delta=delta)

                    # F[n, m, u_i] = p_i
                    # F[n, m, u > u_max] = 0
                    F[n, m, rounded_utility] = total_probability
                    F[n, m, np.ceil(rounded_utility)] = 0
                    G[n, m, utility] = total_probability
    if verbose:
        pprint.pprint(F)

    # Compare to non-rounding version.
    approx = pd.Series(F).to_frame().reset_index()
    approx.columns = ['n', 'm', 'ru', 'p']
    exact = pd.Series(G).to_frame().reset_index()
    exact.columns = ['n', 'm', 'u', 'p']
    check = exact.merge(approx, on=['n', 'm', 'p'])
    check['err'] = (check.ru - check.u).abs()
    check = check[(check.ru != 0)]
    return check.groupby(['n', 'm']).last()
