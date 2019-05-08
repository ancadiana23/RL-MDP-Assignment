import numpy as np
from collections import defaultdict
import sys


def make_epsilon_greedy_policy(action_count: int, q: dict, epsilon=0.0):
    """
    This function creates an epsilon greedy policy based on the given Q.
    :param action_count: Number of actions
    :param q: A dictionary that maps from a state to the action values
    for all possible nA actions (represented as an array)
    :param epsilon: Probability to select a random action
    :param distribute_prob: Whether or not to distribute the probability between best actions
                            or just choose the first best action an assign it all the probability mass.
    :return: A function that takes as argument an observation and returns
             the probabilities of each action.
    """

    def policy_func(observation, eps=epsilon):
        actions = np.ones(action_count, dtype=float) * eps / action_count
        q_values = q[observation]
        best_action = np.argmax(q_values)
        actions[best_action] += 1.0 - eps
        return actions

    return policy_func


def print_in_line(episode_i):
    sys.stdout.write("EPISODE [{0}]   \r".format(episode_i))
    sys.stdout.flush()


# TODO: SAVE the T for episode.
def q_learning(env, num_episodes: int, q=None, discount_factor=1.0, alpha=0.3, epsilon=0.1):
    """
    Q-Learning (off-policy control) algorithm implementation as described in
    http://incompleteideas.net/sutton/book/ebook/node65.html.
    :param env: The OpenAI Env used
    :param num_episodes: Number of episodes to run the algorithm for
    :param q: Q action state values to start from
    :param discount_factor: The gamma discount factor
    :param alpha: The learning rate
    :param epsilon: Chance to sample a random action
    :return: q the optimal value function
    """
    # initialize the action value function
    if q is None:
        q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = None
    # initialize the policy for the Q function case
    policy = make_epsilon_greedy_policy(env.action_space.n, epsilon=epsilon, q=q)
    # loop for each episode
    # tqdm nice for visualization
    # for episode in tqdm(range(num_episodes)):
    for episode in range(num_episodes):
        print_in_line(episode)
        # initialize the state
        state = env.reset()
        done = False
        t = 0
        # loop for each step in the episode
        while not done:
            # choose action from state based on the policy
            action_prob = policy(state)
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            # take a step in the environment
            next_state, reward, done, _ = env.step(action)
            # q learning update for Q function case
            best_next_action = np.argmax(q[next_state])
            q[state][action] += alpha * (
                reward + discount_factor * q[next_state][best_next_action] - q[state][action]
            )
            # check for finished episode
            if done:
                break
            # otherwise update state
            t += 1
            state = next_state
    return q
