from collections import defaultdict

import numpy as np
import sys

import utils


def print_in_line(episode_i):
    sys.stdout.write("EPISODE [{0}]   \r".format(episode_i))
    sys.stdout.flush()


# TODO: SAVE the T for episode.
def q_learning(env, num_episodes: int, q=None, discount_factor=1.0, alpha=0.3, policy=None):
    """
    Q-Learning (off-policy control) algorithm implementation as described in
    http://incompleteideas.net/sutton/book/ebook/node65.html.
    :param env: The OpenAI Env used
    :param num_episodes: Number of episodes to run the algorithm for
    :param q: Q action state values to start from
    :param discount_factor: The gamma discount factor
    :param alpha: The learning rate
    :param policy: The policy to use during training
    :return: q the optimal value function
    """
    # initialize the action value function
    if q is None:
        q = defaultdict(lambda: np.zeros(env.action_space.n))
        policy = utils.make_epsilon_greedy_policy(env.action_space.n, epsilon=0.1, q=q)

    # loop for each episode
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
            # otherwise update state and increase the t
            t += 1
            state = next_state
    return q


def double_q_learning(env, num_episodes: int, q_A=None, q_B=None, discount_factor=1.0, alpha=0.3, policy_A=None, policy_B=None):
    """
    Double Q-Learning (off-policy control) algorithm implementation as described in
    http://incompleteideas.net/sutton/book/ebook/node65.html.
    :param env: The OpenAI Env used
    :param num_episodes: Number of episodes to run the algorithm for
    :param q_A: First Q action state values to start from
    :param q_B: Second Q action state values to start from
    :param discount_factor: The gamma discount factor
    :param alpha: The learning rate
    :param policy_A: First policy to use during training
    :param policy_B: Second policy to use during training
    :return: q the optimal value function
    """
    # initialize the action value functions and policies
    if q_A is None:
        q_A = defaultdict(lambda: np.zeros(env.action_space.n))
        policy_A = utils.make_epsilon_greedy_policy(env.action_space.n, epsilon=0.1, q=q_A)
    if q_B is None:
        q_B = defaultdict(lambda: np.zeros(env.action_space.n))
        policy_B = utils.make_epsilon_greedy_policy(env.action_space.n, epsilon=0.1, q=q_B)
        
    # loop for each episode
    for episode in range(num_episodes):
        print_in_line(episode)
        # initialize the state
        state = env.reset()
        done = False
        t = 0
        # loop for each step in the episode
        while not done:
            # choose action from state based on both policies
            action_prob = (policy_A(state) + policy_B(state)) / 2.0
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            # take a step in the environment
            next_state, reward, done, _ = env.step(action)

            # randomly assign update and target state action functions
            if np.random.choice([0, 1]) == 0:
                q_update = q_A
                q_target = q_B
            else:
                q_update = q_B
                q_target = q_A

            # q learning update for Q function case
            best_next_action = np.argmax(q_update[next_state])
            q_update[state][action] += alpha * (
                reward + discount_factor * q_target[next_state][best_next_action] - q_update[state][action]
            )
            # check for finished episode
            if done:
                break
            # otherwise update state and increase the t
            t += 1
            state = next_state
    return q_A
