import numpy as np
import sys
from gym.envs.toy_text import discrete
from collections import defaultdict

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ROWS = 4
COLUMNS = 4


class GridWorldEnv(discrete.DiscreteEnv):
    """
    Custom GridWorld environment.
    The grid is depicted below:
    o  o  o  T
    o  X  o  X
    o  o  S  X
    R  X  X  X
    The agent starts from R.
    Cracks are represented by X, the shipwrek by S and the terminal state by T.
    Falling into a crack incurs a reward of -10 and finishes the episode.
    Moving into the terminal state incurs a reward of +100 and finishes the episode.
    Moving into the Shipwreck incurs a reward of +20 and the episode continues.
    At each time-step, the robot has a 5% chance of slipping on the ice, and go
    all the way to the side of the environment
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self):
        self.shape = (4, 4)

        state_count: int = np.prod(self.shape)
        action_count = 4

        # crack location
        self._crack = np.zeros(self.shape, dtype=np.bool)
        self._crack[1, 1] = True
        self._crack[1, 3] = True
        self._crack[2, 3] = True
        self._crack[3, 1:4] = True

        self._win = np.zeros(self.shape, dtype=np.bool)
        self._win[0, 3] = True

        self.terminal_states = np.zeros(state_count, dtype=np.bool)

        # Calculate transition probabilities and rewards
        # Also calculate initial state distribution
        transition_prob = {}
        isd = np.zeros(state_count, dtype=np.float)
        for s in range(state_count):
            print(f"calculating transitions {s}")
            # transition probabilities
            position = np.unravel_index(s, self.shape)
            transition_prob[s] = {a: [] for a in range(action_count)}
            transition_prob[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            transition_prob[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            transition_prob[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            transition_prob[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

            if position == (3, 0):
                isd[s] = 1.0
            current_state = np.ravel_multi_index(tuple(position), self.shape)
            if self._crack[tuple(position)] or self._win[tuple(position)]:
                self.terminal_states[current_state] = True
            else:
                self.terminal_states[current_state] = False

        # Calculate actual probabilities for isd
        isd = isd / np.sum(isd)

        super(GridWorldEnv, self).__init__(state_count, action_count, transition_prob, isd)

    def _calculate_transition_prob(self, current, delta):
        """
        Determines the outcome for an action.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position from transition
        :return: (1.0, new_state, reward, done)
        """

        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        is_same_state = tuple(new_position) == current

        new_position_slip = np.array(current) + np.array(delta)
        while new_position_slip[0] in range(4) and new_position_slip[1] in range(4):
            # check to see if we hit a crack
            is_crack = self._crack[tuple(new_position_slip)]
            if is_crack:
                # print(f"I'm in a crack and my new pos = {new_position_slip}")
                break
            new_position_slip = new_position_slip + np.array(delta)
        new_position_slip = self._limit_coordinates(new_position_slip).astype(int)
        new_state_slip = np.ravel_multi_index(tuple(new_position_slip), self.shape)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        shipwreck = (2, 2)
        terminal_state = (self.shape[0] - 4, self.shape[1] - 1)
        is_done_crack = self._crack[tuple(new_position)]
        is_done_crack_slip = self._crack[tuple(new_position_slip)]
        is_done_terminal = tuple(new_position) == terminal_state
        is_done_terminal_slip = tuple(new_position_slip) == terminal_state
        is_shipwreck = tuple(new_position) == shipwreck
        transition_prob = list()
        if is_done_crack:
            return [(1, new_state, -10, True)]
        elif is_done_terminal:
            return [(1, new_state, 100, True)]
        elif is_shipwreck:
            transition_prob.append((0.95, new_state, 20, False))
        elif is_same_state:
            return [(1, new_state, 0, False)]
        else:
            transition_prob.append((0.95, new_state, 0, False))

        if is_done_crack_slip:
            transition_prob.append((0.05, new_state_slip, -10, True))
        elif is_done_terminal_slip:
            transition_prob.append((0.05, new_state_slip, 100, True))
        else:
            transition_prob.append((0.05, new_state_slip, 0, False))

        return transition_prob

    def _limit_coordinates(self, position):
        """
        Prevents the agent from falling of the grid.
        :param position: Current position on the grid as (row, col)
        :return: Adjusted position
        """
        position[0] = max(0, min(position[0], self.shape[0] - 1))
        position[1] = max(0, min(position[1], self.shape[1] - 1))
        return position

    def render(self, mode="human"):
        outfile = sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            elif position == (ROWS - 4, COLUMNS - 1):
                output = " G "
            elif self._crack[position]:
                output = " C "
            elif position == (2, 2):
                output = " S "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

    def render_policy(self, policy):
        """
        Renders the policy of the grid using the given policy function
        :param policy: the policy function given
        """
        outfile = sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            shipwreck = position == (2, 2)
            if position == (ROWS - 4, COLUMNS - 1):
                output = " G "
            elif self._crack[position]:
                output = " C "
            else:
                if type(policy) is defaultdict or type(policy) is dict:
                    action_prob = policy[s]
                else:
                    action_prob = policy(s)
                action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
                if action == 0:
                    output = " \u2191 " if not shipwreck else " S|\u2191 "
                elif action == 1:
                    output = " \u2192 " if not shipwreck else " S|\u2192 "
                elif action == 2:
                    output = " \u2193 " if not shipwreck else " S|\u2193 "
                else:
                    output = " \u2190 " if not shipwreck else " S|\u2190 "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")
