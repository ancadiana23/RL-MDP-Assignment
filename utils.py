import numpy as np
import sys


def print_in_line(episode_i, max_episodes):
    episode_i += 1
    episode = np.round(episode_i * 100 / max_episodes)
    filling = "#" * int(episode)
    sys.stdout.write("{0}% - [{1}]   \r".format(episode, filling))
    sys.stdout.flush()
    if episode_i == max_episodes:
        sys.stdout.write("[{0}] - DONE   \r".format(filling))
        sys.stdout.flush()
        print("")


def make_epsilon_greedy_policy(action_count: int, q: dict, epsilon=0.0):
    """
    This function creates an epsilon greedy policy based on the given Q.
    :param action_count: Number of actions
    :param q: A dictionary that maps from a state to the action values
    for all possible nA actions (represented as an array)
    :param epsilon: Probability to select a random action
    :return: A function that takes and observation and output 
        probabilities of each action.
    """

    def policy_func(observation, eps=epsilon):
        actions = np.ones(action_count, dtype=float) * eps / action_count
        q_values = q[observation]
        best_action = np.argmax(q_values)
        actions[best_action] += 1.0 - eps
        return actions

    return policy_func


def make_softmax_policy(action_count: int, q: dict, temperature=1.0):
    """
    This function creates a softmax policy based on the given Q.
    :param action_count: Number of actions
    :param q: A dictionary that maps from a state to the action values
        for all possible nA actions (represented as an array)
    :param temperature: temperature parameter used in the Boltzmann distribution
    :return: A function that takes and observation and output probabilities 
        of each action.
    """
    if temperature == 0.0:
        print("Temperature has to be different than 0.0.")
        return None

    def policy_func(observation, temperature=temperature):
        q_values = np.array(q[observation])
        exp_values = np.exp(q_values / temperature)
        actions = exp_values / sum(exp_values)
        return actions

    return policy_func
