import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import *

HOME = os.environ['USER'] + '/cap_fptas/'
plt.style.use('seaborn-whitegrid')
os.chdir(HOME)


__author__ = 'kqureshi'


class CAP:

    def __init__(self):

        self.T = 0.9
        self.K = 50
        self.eps = 0.01

# Disable



def v_prime(V: float, p: float, u: float) -> float:
    """
    The expected utility required in the subset prior to new addition
    :param V:
    :param p:
    :param u:
    :return:
    """
    return (V - (p * u)) / (1 - p)


def get_vstar(prob: pd.DataFrame,
              T: float,
              K: int,
              eps: float) -> float:
    """
    (1) Add a safety school to satisfy the risk requirements
    (2) Add a reach school
    :param prob:
    :param T:
    :param K:
    :param eps:
    :return:
    """

    # Constrain by risk, sort by utility, and select the first safety school
    select = prob[prob.PROBABILITY >= T].sort_values('UTILITY', ascending=False).reset_index(drop=True)
    if select.shape[1] < 6:
        num, _, p_c, u_c, eu_c = select.iloc[0]
    else:
        num, _, p_c, u_c, eu_c, ru_c = select.iloc[0]
    school_ids = [num]

    # Initialize the risk and expected utility
    F, V = p_c, eu_c
    if select.shape[1] < 6:
        num, _, p_c, u_c, eu_c = prob.iloc[0]
    else:
        num, _, p_c, u_c, eu_c, ru_c = prob.iloc[0]

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
    """
    Use trimming/rounding of probabilities forthe FPTAS
    :param prob:
    :param K:
    :param V_star:
    :param eps:
    :return:
    """

    # Define bin size
    delta = eps * V_star / K
    print('Delta is {}'.format(delta))

    # Rounded utility
    prob['RU'] = np.round(prob.UTILITY / delta) * delta

    # Error bound
    error = delta * K, eps * V_star
    print('Total error bounded in {}'.format(error))
    return prob


def get_value_function(prob: pd.DataFrame,
                       T: float,
                       K: int,
                       eps: float = 0.01) -> List[float]:
    """

    :param prob:
    :param T:
    :param K:
    :param eps:
    :return:
    """

    # Calculate V*
    V_star = get_vstar(prob=prob, T=T, K=K, eps=eps)

    # Trim probability
    prob = trim_probabilities(prob=prob, K=K, V_star=V_star, eps=eps)
    schools, utility, rejection_probs, admission_probs = [], [], [], []
    V = V_star
    max_prob = prob.PROBABILITY.max()

    # Case: A single schol cannot satisfy the risk requirement, for example 99.99%
    if max_prob >= T:
        tray = prob[prob.PROBABILITY >= T].sort_values(['PROBABILITY', 'UTILITY'], ascending=[True, False])
        safety_school = tray.iloc[0]
        num, sid, p, u, e, re = safety_school.values
        schools.append(num)
        utility.append(re)

        # Update rejection/admission probability
        rejection_probs.append(1 - p)
        acceptance = 1 - np.product(rejection_probs)
        print('Added safety school {}. Current acceptance probability {}'.format(num, acceptance))
        admission_probs.append(acceptance)

    # Until the number of required schools is reached, run CAP algorithm
    j = 0
    while len(schools) < K:
        risky = prob.sort_values('UTILITY', ascending=False)
        num, sid, p, u, e, re = risky.iloc[j].values
        schools.append(num)
        rejection_probs.append(1 - p)

        # Update V' for every new school
        vprime = (V - (p * re)) / (1 - p)
        utility.append((p * re) + ((1 - p) * vprime))

        # Update acceptance/rejection probability for every new school
        acceptance = 1 - np.product(rejection_probs)
        admission_probs.append(acceptance)
        print('Added risky school {}. Current probability {}. Current V {}'.format(num, acceptance,
                                                                                   (p * re) + ((1 - p) * vprime)))
        j += 1
        # return schools
    return admission_probs


def run_value_function(prob: pd.DataFrame,
                       T: float,
                       K: int,
                       eps: float,
                       utility_power: Optional[List[float]] = None,
                       file_name: str = None,
                       plot: bool = True) -> None:
    """

    :param prob:
    :param T:
    :param K:
    :param eps:
    :param utility_power:
    :param file_name:
    :param plot:
    :return:
    """
    print('Running FPTAS for T={}, K={}'.format(T, K))

    if 'RU' in list(prob.columns):
        prob = prob.drop('RU', axis=1)
    if isinstance(utility_power, list):
        blockPrint()
        for j in utility_power:

            # Assume higher utility for lower probability
            prob['UTILITY'] = 1 / (prob.PROBABILITY) ** (j)
            prob.UTILITY += np.random.randn(len(prob.UTILITY)) / 10

            # Sort by expected utility
            prob['EU'] = prob.PROBABILITY * prob.UTILITY
            prob = prob.sort_values('EU', ascending=False).reset_index(drop=True)
            if 'RU' in list(prob.columns):
                prob = prob.drop('RU', axis=1)

            # Update V*
            v_star = get_vstar(prob=prob, T=T, K=K, eps=eps)
            print('V* is {}'.format(v_star))

            # Trim probabilities
            tprob = trim_probabilities(prob=prob, K=K, V_star=v_star, eps=eps)

            # Update value function
            value_function = get_value_function(prob=tprob, T=T, K=K)
            if not plot: return value_function
            plt.plot(value_function, label='n={}'.format(float(j)))

        plt.ylabel('Probability')
        plt.xlabel('Num Schools')
        plt.legend()
        if isinstance(file_name, str):
            plt.savefig(HOME + '/figures/{}.pdf'.format(file_name), format='pdf')
        plt.show()
        enablePrint()

    else:

        # Update V*
        v_star = get_vstar(prob=prob, T=T, K=K, eps=eps)
        print('V* is {}'.format(v_star))

        # Trim probabilities
        tprob = trim_probabilities(prob=prob, K=K, V_star=v_star, eps=eps)

        # Update value function
        value_function = get_value_function(prob=tprob, T=T, K=K)
        if not plot: return value_function
        plt.plot(value_function)
        plt.ylabel('Probability')
        plt.xlabel('Num Schools')
        if isinstance(file_name, str):
            plt.savefig('/Users/kq/Documents/cap_fptas/figures/{}.pdf'.format(file_name), format='pdf')
        plt.show()


def read_probabilities() -> pd.DataFrame:
    """

    :return:
    """

    prob = pd.read_csv(HOME + '/probabilities.csv')

    # U = 1 / p^2 + eps
    prob['UTILITY'] = 1 / (prob.PROBABILITY) ** (2)
    prob.UTILITY += np.random.randn(len(prob.UTILITY)) / 10
    prob['EU'] = prob.PROBABILITY * prob.UTILITY

    # Sort by expected utility
    prob = prob.sort_values('EU', ascending=False).reset_index(drop=True)
    prob = prob[(prob.PROBABILITY > 0) & (prob.PROBABILITY < 1)].reset_index(drop=True)
    prob.insert(0, 'NUMBER', range(len(prob)))
    return prob


def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore

def enablePrint():
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    cap = CAP()
    prob = read_probabilities()
    run_value_function(prob=prob, T=cap.T, K=cap.K, eps=cap.eps)

