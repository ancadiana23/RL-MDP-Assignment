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
    :param epsilon: Chance to sample a random action
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
