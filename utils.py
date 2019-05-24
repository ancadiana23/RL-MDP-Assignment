import numpy as np
import sys
import matplotlib.pyplot as plt


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


def print_in_line(episode_i, max_episodes):
    episode_i += 1
    episode = np.round(episode_i * 100 / max_episodes)
    filling = "#" * int(episode)
    sys.stdout.write("{0}% - [{1}]   \r".format(episode, filling))
    sys.stdout.flush()
    if episode_i == max_episodes:
        print("")


def plot_episode_stats(iterations, rewards, title, label, label_data, smooth=10):
    """
    Plots the episode statistics
    :param stats: the episode statistics
    :param smooth: smoothing window
    """
    plt.style.use("seaborn-darkgrid")

    # create a color palette
    palette = plt.get_cmap("Set1")

    # Plot the episode length over time
    plt.figure(figsize=(10, 5))
    for i, iteration in enumerate(iterations):
        plt.plot(
            iteration,
            marker="",
            color=palette(i + 1),
            linewidth=1.9,
            alpha=0.9,
            label=f"{label}={label_data[i]}",
        )
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    fn = f"{title} Episode Length over Time"
    plt.title(f"{title} Episode Length over Time")
    plt.legend()

    plt.savefig(fn)

    # Plot the episode reward over time
    plt.figure(figsize=(10, 5))
    for i, reward in enumerate(rewards):
        box = np.ones(smooth) / smooth
        rewards_smoothed = np.convolve(reward, box, mode="same")
        plt.plot(
            rewards_smoothed,
            marker="",
            color=palette(i + 1),
            linewidth=1.9,
            alpha=0.9,
            label=f"{label}={label_data[i]}",
        )
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    fn = f"{title} Episode Reward over Time"
    plt.title(f"{title} Episode Reward over Time")
    plt.legend()
    plt.savefig(fn)

    # Plot time steps and episode number
    plt.figure(figsize=(10, 5))
    for i, iteration in enumerate(iterations):
        plt.plot(
            np.cumsum(iteration),
            np.arange(len(iteration)),
            marker="",
            color=palette(i + 1),
            linewidth=1.9,
            alpha=0.9,
            label=f"{label}={label_data[i]}",
        )
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    fn = f"{title} Episode per time step"
    plt.title(f"{title} Episode per time step")
    plt.legend()

    plt.savefig(fn)

    # plt.show()


def make_epsilon_greedy_policy(action_count: int, q: dict, epsilon=0.0):
    """
    This function creates an epsilon greedy policy based on the given Q.
    :param action_count: Number of actions
    :param q: A dictionary that maps from a state to the action values
    for all possible nA actions (represented as an array)
    :param epsilon: Probability to select a random action
    :return: A function that takes and observation and output probabilities of each action.
    """

    def randargmax(b, **kw):
        """ a random tie-breaking argmax"""
        return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)

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
    :return: A function that takes and observation and output probabilities of each action.
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
