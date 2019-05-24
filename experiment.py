from gridworld import GridWorldEnv

import dp
import learnalg
import ps

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import utils
import time


# Environment
env = GridWorldEnv()


#  POLICY EVALUATION
print("Policy evaluation using policy evaluation")
state_values, _ = dp.policy_evaluation(env=env, discount_factor=0.9)
print("Value States-> \n", state_values)
# print_state_latex(state_values)

# VALUE ITERATION
policy, state_values, deltas_value, t = dp.value_iteration(env, discount_factor=0.9)

print("Optimal policy found using Value Iteration algorithm  [0.9] found in ")
print("Elapsed time: %.5f [sec]" % (t[0]))
print("CPU elapsed time: %.5f [sec]" % (t[1]))
env.render_policy(policy=policy)
print(state_values)

# POLICY ITERATION
policy, state_values, deltas_iteration, t = dp.policy_iteration(env=env, discount_factor=0.9)
print("Optimal policy found using Policy Iteration algorithm [0.9] found in ")
print("Elapsed time: %.5f [sec]" % (t[0]))
print("CPU elapsed time: %.5f [sec]" % (t[1]))
env.render_policy(policy=policy)
print("Value States -> \n", state_values)

# SIMPLE POLICY ITERATION
policy, state_values, deltas_iteration, t = dp.policy_iteration(env=env, discount_factor=0.9, simple=True)
print("Optimal policy found using Simple Policy Iteration algorithm [0.9] found in ")
print("Elapsed time: %.5f [sec]" % (t[0]))
print("CPU elapsed time: %.5f [sec]" % (t[1]))
env.render_policy(policy=policy)
print("Value States -> \n", state_values)

# PARAMETER SWEEPING
t2_start = time.perf_counter()
t2_cpu_start = time.process_time()
policy = ps.prioritized_sweeping(env=env, discount_factor=0.9)
t2_stop = time.perf_counter()
t2_cpu_stop = time.process_time()
time_elapsed_2 = t2_stop - t2_start
time_cpu_elapsed_2 = t2_cpu_stop - t2_cpu_start
env.render_policy(policy=policy)
print("Time elapsed {} and time elapsed cpu {}".format(time_elapsed_2, time_cpu_elapsed_2))


# HYPER PARAMETERS
alpha = 1e-3
num_iterations = 1000
# epsilons = [0.01, 0.1, 0.3, 0.5]
epsilons = [0.1]
# temperature = [0.5, 1.2, 2, 5]
temperature = [1.2]


# SARSA
i = list()
r = list()
for epsilon in epsilons:
    q = defaultdict(lambda: np.zeros(env.action_space.n))
    train_policy = utils.make_epsilon_greedy_policy(env.action_space.n, epsilon=epsilon, q=q)
    (q_sarsa, iterations, rewards) = learnalg.sarsa(
        env, num_iterations, alpha=alpha, discount_factor=0.9, policy=train_policy, q=q
    )
    policy = utils.make_epsilon_greedy_policy(epsilon=0.0, action_count=env.action_space.n, q=q_sarsa)
    print("SARSA e=", epsilon)
    for item in sorted(q_sarsa.keys()):
        print(f"state {item} - Actions {q_sarsa[item]}")
    env.render_policy(policy=policy)
    i.append(iterations)
    r.append(rewards)


# Q-Learning with Softmax policy
i = list()
r = list()
for temp in temperature:
    q = defaultdict(lambda: np.zeros(env.action_space.n))
    train_policy = utils.make_softmax_policy(env.action_space.n, temperature=temp, q=q)
    (q, iterations, rewards) = learnalg.q_learning(
        env, num_iterations, alpha=alpha, discount_factor=0.9, q=q, policy=train_policy
    )
    policy = utils.make_epsilon_greedy_policy(epsilon=0, action_count=env.action_space.n, q=q)
    print("Softmax exploration Q-Learning temp=", temp)
    for item in sorted(q.keys()):
        print(f"state {item} - Actions {q[item]}")
    env.render_policy(policy=policy)
    i.append(iterations)
    r.append(rewards)


# Q-Learning with Epsilon greedy exploration
i = list()
r = list()
for epsilon in epsilons:
    q = defaultdict(lambda: np.zeros(env.action_space.n))
    train_policy = utils.make_epsilon_greedy_policy(env.action_space.n, epsilon=epsilon, q=q)
    (q, iterations, rewards) = learnalg.q_learning(
        env, num_iterations, alpha=alpha, discount_factor=0.9, policy=train_policy, q=q
    )
    policy = utils.make_epsilon_greedy_policy(epsilon=0, action_count=env.action_space.n, q=q)
    print("Epsilon greedy Q-Learning e= ", epsilon)
    for item in sorted(q.keys()):
        print(f"state {item} - Actions {q[item]}")
    env.render_policy(policy=policy)
    i.append(iterations)
    r.append(rewards)

# Double Q learning Epsilon Greedy
i = list()
r = list()
for epsilon in epsilons:
    q_a = defaultdict(lambda: np.zeros(env.action_space.n))
    policy_a = utils.make_epsilon_greedy_policy(env.action_space.n, epsilon=epsilon, q=q_a)
    q_b = defaultdict(lambda: np.zeros(env.action_space.n))
    policy_b = utils.make_epsilon_greedy_policy(env.action_space.n, epsilon=epsilon, q=q_b)
    (q_A, iterations, rewards) = learnalg.double_q_learning(
        env,
        num_iterations,
        alpha=alpha,
        discount_factor=0.9,
        policy_A=policy_a,
        q_A=q_a,
        policy_B=policy_b,
        q_B=q_b,
    )
    policy = utils.make_epsilon_greedy_policy(epsilon=0.0, action_count=env.action_space.n, q=q_A)
    print("Double Q_Learning e=", epsilon)
    for item in sorted(q_A.keys()):
        print(f"state {item} - Actions {q_A[item]}")
    env.render_policy(policy=policy)
    i.append(iterations)
    r.append(rewards)


# ELEGIBILITY TRACES
q, iteations, rewards = learnalg.q_learning_with_eligibility_traces(
    env, num_iterations, alpha=alpha, eligibility_factor=0.2, discount_factor=0.9
)
policy = utils.make_epsilon_greedy_policy(epsilon=0.0, action_count=env.action_space.n, q=q)
print("Q-Learning with eligibility traces")
for item in sorted(q.keys()):
    print(f"state {item} - Actions {q[item]}")
env.render_policy(policy=policy)


# HYPER PARAMETERS FOR EXPERIENCE REPLAY
num_iterations = 200
alpha = 1e-3
T = 2  # Length of each trajectory
N = 100  # Number of replays

# Best params -> num_iteration = 200, alpha = 1e-3, T = 2, N = 100

# Experience Replay Q-Learning
i = list()
r = list()
for epsilon in epsilons:
    q = defaultdict(lambda: np.zeros(env.action_space.n))
    train_policy = utils.make_epsilon_greedy_policy(env.action_space.n, epsilon=epsilon, q=q)
    (q, iterations, rewards) = learnalg.q_learning_experience(
        env, num_iterations, alpha=alpha, discount_factor=0.9, T=T, N=N, policy=train_policy, q=q
    )
    policy = utils.make_epsilon_greedy_policy(epsilon=0.0, action_count=env.action_space.n, q=q)
    print("Q-Learning with experience relay")
    for item in sorted(q.keys()):
        print(f"state {item} - Actions {q[item]}")
    env.render_policy(policy=policy)
    i.append(iterations)
    r.append(rewards)
