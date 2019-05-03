import dp
import ps
from gridworld import GridWorldEnv
import time

env = GridWorldEnv()

print("Policy evaluation using policy evaluation")
state_values = dp.policy_evaluation(env=env, discount_factor=0.9)
print(state_values)

policy, state_values = dp.value_iteration(env, discount_factor=0.9)

print("Optimal policy found using Value Iteration algorithm")
env.render_policy(policy=policy)
print(state_values)


policy, state_values = dp.policy_iteration(env=env, discount_factor=0.9)
print("Optimal policy found using Policy Iteration algorithm")
env.render_policy(policy=policy)
print(state_values)

t1_start = time.perf_counter()
t1_cpu_start = time.process_time()
policy, state_values = dp.policy_iteration(env=env, discount_factor=0.9, simple=True)
t1_stop = time.perf_counter()
t1_cpu_stop = time.process_time()
time_elapsed_1 = t1_stop - t1_start
time_cpu_elapsed_1 = t1_cpu_stop - t1_cpu_start
print("Optimal policy found using Simple Policy Iteration algorithm")
env.render_policy(policy=policy)
print(state_values)
print("Time elapsed {} and time elapsed cpu {}".format(time_elapsed_1, time_cpu_elapsed_1))

t2_start = time.perf_counter()
t2_cpu_start = time.process_time()
policy = ps.prioritized_sweeping(env=env, discount_factor=0.9)
t2_stop = time.perf_counter()
t2_cpu_stop = time.process_time()
time_elapsed_2 = t2_stop - t2_start
time_cpu_elapsed_2 = t2_cpu_stop - t2_cpu_start
env.render_policy(policy=policy)

print("Time elapsed {} and time elapsed cpu {}".format(time_elapsed_2, time_cpu_elapsed_2))