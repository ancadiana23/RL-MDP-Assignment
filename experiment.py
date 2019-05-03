import dp
import ps
from gridworld import GridWorldEnv
import time
import numpy as np

import matplotlib.pyplot as plt


def print_state_latex(states):
    print(
        "\\multicolumn{1}{|l|}{\\textbf{0}}"
        + f"& {np.around(states[0],5)}    & {np.around(states[1],5)}      & {np.around(states[2],5)}      & {np.around(states[3],5)}         \\\\ \\hline"
    )
    print(
        "\\multicolumn{1}{|l|}{\\textbf{0}}"
        + f"& {np.around(states[4],5)}    & {np.around(states[5],5)}      & {np.around(states[6],5)}      & {np.around(states[7],5)}         \\\\ \\hline"
    )
    print(
        "\\multicolumn{1}{|l|}{\\textbf{0}}"
        + f"& {np.around(states[8],5)}    & {np.around(states[9],5)}      & {np.around(states[10],5)}       & {np.around(states[11],5)}          \\\\ \\hline"
    )
    print(
        "\\multicolumn{1}{|l|}{\\textbf{0}}"
        + f"& {np.around(states[12],5)}     & {np.around(states[13],5)}      & {np.around(states[14],5)}       & {np.around(states[15],5)}          \\\\ \\hline"
    )


def plot_stuff(deltas_value, deltas_iteration):
    # Initialize the figure
    plt.style.use("seaborn-darkgrid")

    # create a color palette
    palette = plt.get_cmap("Set1")

    # multiple line plot
    # Find the right spot on the plot
    plt.subplot(3, 1, 1)

    # Plot the lineplot
    plt.plot(deltas_value, marker="", color=palette(1), linewidth=1.9, alpha=0.9, label="Value Iteration")
    # plt.tick_params(labelbottom=False)
    # Add title
    plt.title("Value Iteration", loc="left", fontsize=12, fontweight=0, color=palette(1))

    plt.subplot(3, 1, 2)

    # Plot the lineplot
    plt.plot(
        deltas_iteration[0], marker="", color=palette(2), linewidth=1.9, alpha=0.9, label="Policy Iteration"
    )
    # plt.tick_params(labelleft=False)
    # plt.tick_params(labelbottom=False)
    # Add title
    plt.title("Policy Iteration - iteration 1", loc="left", fontsize=12, fontweight=0, color=palette(2))

    plt.subplot(3, 1, 3)

    # Plot the lineplot
    plt.plot(
        deltas_iteration[1], marker="", color=palette(3), linewidth=1.9, alpha=0.9, label="Policy Iteration"
    )
    # plt.tick_params(labelleft=False)
    # Add title
    plt.title("Policy Iteration - iteration 2", loc="left", fontsize=12, fontweight=0, color=palette(3))
    # Same limits for everybody!
    # plt.xlim(0,10)
    # plt.ylim(-2,22)

    # Not ticks everywhere
    # if num in range(7) :
    #     plt.tick_params(labelbottom='off')
    # if num not in [1,4,7] :
    #     plt.tick_params(labelleft='off')

    # general title
    plt.suptitle("Delta-Iterations comparation", fontsize=13, fontweight=0, color="black", style="italic")
    plt.tight_layout()
    plt.show()
    # Axis title
    # plt.text(0.5, 0.02, 'Time', ha='center', va='center')
    # plt.text(0.06, 0.5, 'Note', ha='center', va='center', rotation='vertical')

env = GridWorldEnv()

print("Policy evaluation using policy evaluation")
state_values, _ = dp.policy_evaluation(env=env, discount_factor=0.9)
print("Value States-> \n", state_values)
# print_state_latex(state_values)

policy, state_values, deltas_value, t = dp.value_iteration(env, discount_factor=0.9)

print("Optimal policy found using Value Iteration algorithm  [0.9] found in ")
print("Elapsed time: %.5f [sec]" % (t[0]))
print("CPU elapsed time: %.5f [sec]" % (t[1]))
env.render_policy(policy=policy)
print(state_values)

# policy, state_values, deltas_value, t = dp.value_iteration(env, discount_factor=0.6)

# print("Optimal policy found using Value Iteration algorithm [0.6] found in ")
# print("Elapsed time: %.5f [sec]" % (t[0]))
# print("CPU elapsed time: %.5f [sec]" % (t[1]))
# env.render_policy(policy=policy)
# print(state_values)
# # print_state_latex(state_values)

# policy, state_values, deltas_value, t = dp.value_iteration(env, discount_factor=0.1)

# print("Optimal policy found using Value Iteration algorithm [0.1] found in ")
# print("Elapsed time: %.5f [sec]" % (t[0]))
# print("CPU elapsed time: %.5f [sec]" % (t[1]))
# env.render_policy(policy=policy)
# print(state_values)
# # print_state_latex(state_values)


policy, state_values, deltas_iteration, t = dp.policy_iteration(env=env, discount_factor=0.9)
print("Optimal policy found using Policy Iteration algorithm [0.9] found in ")
print("Elapsed time: %.5f [sec]" % (t[0]))
print("CPU elapsed time: %.5f [sec]" % (t[1]))
env.render_policy(policy=policy)
print("Value States -> \n", state_values)

# policy, state_values, deltas_iteration, t = dp.policy_iteration(env=env, discount_factor=0.6)
# print("Optimal policy found using Policy Iteration algorithm [0.6] found in ")
# print("Elapsed time: %.5f [sec]" % (t[0]))
# print("CPU elapsed time: %.5f [sec]" % (t[1]))
# env.render_policy(policy=policy)
# print("Value States -> \n", state_values)
# print_state_latex(state_values)

# policy, state_values, deltas_iteration, t = dp.policy_iteration(env=env, discount_factor=0.1)
# print("Optimal policy found using Policy Iteration algorithm [0.1] found in ")
# print("Elapsed time: %.5f [sec]" % (t[0]))
# print("CPU elapsed time: %.5f [sec]" % (t[1]))
# env.render_policy(policy=policy)
# print("Value States -> \n", state_values)
# print_state_latex(state_values)

#plot_stuff(deltas_value, deltas_iteration)

policy, state_values, deltas_iteration, t = dp.policy_iteration(env=env, discount_factor=0.9, simple=True)
print("Optimal policy found using Simple Policy Iteration algorithm [0.9] found in ")
print("Elapsed time: %.5f [sec]" % (t[0]))
print("CPU elapsed time: %.5f [sec]" % (t[1]))
env.render_policy(policy=policy)
print("Value States -> \n", state_values)


t2_start = time.perf_counter()
t2_cpu_start = time.process_time()
policy = ps.prioritized_sweeping(env=env, discount_factor=0.9)
t2_stop = time.perf_counter()
t2_cpu_stop = time.process_time()
time_elapsed_2 = t2_stop - t2_start
time_cpu_elapsed_2 = t2_cpu_stop - t2_cpu_start
env.render_policy(policy=policy)

print("Time elapsed {} and time elapsed cpu {}".format(time_elapsed_2, time_cpu_elapsed_2))