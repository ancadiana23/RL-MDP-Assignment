from collections import defaultdict, deque

import numpy as np
import sys
import random
import utils


# TODO: SAVE the T for episode.
def q_learning(env, num_episodes: int, q=None, discount_factor=0.9, alpha=0.3, policy=None):
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
        utils.print_in_line(episode, num_episodes)
        # initialize the state
        state = env.reset()
        done = False
        t = 0
        # loop for each step in the episode
        while not done:
            # env.render()
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


def q_learning_experience(
    env, num_episodes: int, q=None, discount_factor=0.9, alpha=0.3, T=2, N=100, policy=None
):
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
    # T = 2  # Length of each trajectory
    # N = 100  # Number of replays
    l = 1  # trajectories index
    # initialize the action value function
    if q is None:
        q = defaultdict(lambda: np.zeros(env.action_space.n))
        policy = utils.make_epsilon_greedy_policy(env.action_space.n, epsilon=0.2, q=q)
        d = deque(maxlen=N)
    # loop for each episode
    for episode in range(num_episodes):
        utils.print_in_line(episode, num_episodes)
        # initialize the state
        state = env.reset()
        done = False
        t = 0
        # loop for each step in the episode
        while not done:
            # env.render()
            # choose action from state based on the policy
            action_prob = policy(state)
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            # take a step in the environment
            next_state, reward, done, _ = env.step(action)
            # q learning update for Q function case
            best_next_action = np.argmax(q[next_state])
            tuple_q = (state, action, reward, next_state, best_next_action)
            d.append(tuple_q)
            # check for finished episode
            if done:
                break
            # otherwise update state and increase the t
            t += 1
            state = next_state
        if episode == l * T:
            for _ in range(N * l * T):
                (state, action, reward, next_state, best_next_action) = random.sample(d, 1)[0]
                q[state][action] += alpha * (
                    reward + discount_factor * q[next_state][best_next_action] - q[state][action]
                )
            l += 1
    return q


def double_q_learning(env, num_episodes: int, q_A=None, q_B=None, discount_factor=0.9, alpha=0.3, policy_A=None, policy_B=None):
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
        utils.print_in_line(episode, num_episodes)
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


def sarsa(env, num_episodes: int, q=None, discount_factor=1.0, alpha=0.3, policy=None):
    """
    Sarsa (on-policy TD control) algorithm implementation as described in
    http://incompleteideas.net/sutton/book/ebook/node64.html.
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
        utils.print_in_line(episode, num_episodes)
        # initialize the state
        state = env.reset()
        done = False
        t = 0
        # loop for each step in the episode
        # choose action from state based on the policy
        action_prob = policy(state)
        action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        while not done:
            # take a step in the environment using a
            next_state, reward, done, _ = env.step(action)
            # Choose action from next_state based on the policy
            next_action_prob = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_prob)), p=next_action_prob)
            # Sarsa update for Q function
            q[state][action] += alpha * (
                reward + discount_factor * q[next_state][next_action] - q[state][action]
            )
            # check for finished episode
            if done:
                break
            # otherwise update state, action and increase the t
            t += 1
            state = next_state
            action = next_action
    return q
