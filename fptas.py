import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import *

plt.style.use('seaborn-whitegrid')

def v_prime(V: float, p: float, u: float) -> float:
    return (V - (p * u)) / (1 - p)

def get_vstar(prob: pd.DataFrame,
              T: float,
              K: int,
              eps: float) -> float:
    
    select = prob[prob.PROBABILITY >= T].sort_values('UTILITY', ascending=False).reset_index(drop=True)
    num, _, p_c, u_c, eu_c = select.iloc[0]
    school_ids = [num]
    F = p_c
    V = eu_c
    num, _, p_c, u_c, eu_c = prob.iloc[0]
    # Assign the probability to F based on expected utility
    F_step = F + p_c if V <= eu_c else F

    # Add school if expected utility increases
    if F_step > 0:
        school_ids.append(num)

    # V' is the utility of the schools before the latest addition    
    g_c = v_prime(V=eu_c, p=p_c, u=u_c)

    # Compare probability before and after potential addition
    F_new = max(F, p_c + ((1 - p_c) * F_step))
    V_star = V + eu_c
    return V_star

def trim_probabilities(prob: pd.DataFrame, 
                       K: int,
                       V_star: float,
                       eps: float) -> pd.DataFrame:

    delta = eps * V_star / K
    print('Delta is {}'.format(delta))
    prob['RU'] = np.round(prob.UTILITY / delta) * delta
    error = delta * K, eps * V_star
    print('Total error bounded in {}'.format(error))
    return prob

def get_value_function(prob: pd.DataFrame,
                       T: float,
                       K: int) -> List[float]:
    
    schools, utility, rejection_probs, admission_probs = [], [], [], []
    V = V_star
    max_prob = prob.PROBABILITY.max()

    if max_prob >= T:
        tray = prob[prob.PROBABILITY >= T].sort_values(['PROBABILITY', 'UTILITY'], ascending=[True,False])
        safety_school = tray.iloc[0]
        num, sid, p, u, e, re = safety_school.values
        schools.append(num)
        utility.append(re)
        rejection_probs.append(1-p)
        acceptance = 1 - np.product(rejection_probs)
        print('Added safety school {}. Current acceptance probability {}'.format(num, acceptance))
        admission_probs.append(acceptance)
    j = 0
    while len(schools) < K:
        risky = prob.sort_values('UTILITY', ascending=False)
        num, sid, p, u, e, re = risky.iloc[j].values
        schools.append(num)
        rejection_probs.append(1-p)
        vprime = (V - (p * re)) / (1-p)
        utility.append((p * re) + ((1-p) * vprime))
        acceptance = 1 - np.product(rejection_probs)
        admission_probs.append(acceptance)
        print('Added risky school {}. Current probability {}. Current V {}'.format(num, acceptance, (p * re) + ((1-p) * vprime)))
        j+=1    
    return admission_probs

def run_value_function(prob: pd.DataFrame,
                       T: float,
                       K: int, 
                       eps: float) -> None:
    print('Running FPTAS for T={}, K={}'.format(T,K))
    if 'RU' in list(prob.columns):
        prob = prob.drop('RU', axis=1)
    v_star = get_vstar(prob=prob, T=T, K=K, eps=eps)
    print('V* is {}'.format(v_star))
    tprob = trim_probabilities(prob=prob, K=K, V_star=v_star, eps=eps)
    value_function = get_value_function(prob=tprob, T=T, K=K)
    plt.plot(value_function)
    plt.title('Value Function vs. Number of Schools')
    plt.ylabel('Probability')
    plt.xlabel('Num Schools')
    plt.show()
