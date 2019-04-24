import dp
from gridworld import GridWorldEnv
import lp


env = GridWorldEnv()
lp.linear_programming(env, discount_factor=0.9)
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
