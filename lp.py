"""
 - linear programming
"""
import numpy as np
from collections import defaultdict
from dp import get_action_values
import pulp as plp


def linear_programming(env, discount_factor=1.0):
    """
    Performs linear programming as described in https://arxiv.org/ftp/arxiv/papers/1302/1302.4971.pdf.
    :param env: The OpenAI Gym environment:
                - env.P - transition probabilities of the environment
                - env.P[state][action] - a list of transition tuples
                - env.observation_space.n - number of states
                - env.action_space.n - number of actions
    :param discount_factor: gamma the discount factor
    :return:
    """
    # The primal linear program involves maximizing the sum of the v(s) subject to
    # v(i) <= R(s,a) + gamma sum(p(s,a) x v(s))
    # The intuition here is that, for each state i, the optimal
    # total cost from i is no greater than what would be
    # achieved by first taking action k, for each k E nA.

    # minimize the sum of value(state,a), r(state,a)
    # contrained to value(state,a) = 1 + gamma x prob(state,a) value(state,a)

    # def constraint(x, p):
    #     result = list()
    #     for value in p:
    #         result.append(value[0] * x if len(value) == 1 else value[0] * x + value[1] * x)
    #     return result

    opt_model = plp.LpProblem("I hate you", plp.LpMinimize)

    state_values = np.zeros(env.observation_space.n, dtype=np.float64)

    # for state in env.P.keys():
    #     for a in env.P[state].keys():
    #         value_cost = 0
    #         for prob, _, reward, _ in env.P[state][a]:
    #             best_reward = max([reward for prob, next_state, reward, _ in env.P[state][a]])
    #             value_cost += prob * (best_reward - reward)  # C_state_action
    #         c.append(value_cost)
    #     for a in env.P[state].keys():
    #         p = list()
    #         for prob, _, reward, _ in env.P[state][a]:
    #             p.append()
    #         c.append(value_cost)
    n = 16
    m = 4
    set_I = range(0, n)
    set_J = range(0, m)
    c = {(i, j): 0 for i in set_I for j in set_J}
    p = {(i, j, ni): list() for i in set_I for j in set_J for ni in set_I}

    # C = list()
    # B = list()
    for state in env.P.keys():
        # b = list()
        # c = list()
        for a in env.P[state].keys():
            value_cost = 0
            for prob, next_state, reward, _ in env.P[state][a]:
                # best_reward = max([reward for prob, next_state, reward, _ in env.P[state][a]])
                value_cost += prob * -reward  # C_state_action
                p[(state, a, next_state)].append(prob)
            # if len(p[(state, a, next_state)]) == 1:
            #     p[(state, a, next_state)].append(0)
            c[(state, a)] = value_cost

    p = {(i, j, ni): p[i, j, ni] if p[i, j, ni] else [0] for i in set_I for j in set_J for ni in set_I}

    #     B.append(b)
    #     C.append(c)
    # B = np.ravel(B)
    # C = np.ravel(C)

    # # Minimize sum of X*C
    # fun = lambda x: np.sum(x * C)

    # # Contraints
    # cond = {"type": "eq", "fun": lambda x: 1 + np.dot(discount_factor, constraint(x[0:4], B)) - x[0:4]}

    # b = np.array(b)
    # # A_eq = np.array(A)
    # # res = linprog(c, A_eq=A_eq, b_eq=b)

    # res = opt.minimize(fun, np.random.randint(10,64), constraints=cond)
    # state_values[state] = res.x

    # if x is Continuous
    x_vars = {
        (i, j): plp.LpVariable(cat=plp.LpInteger, name="x_{0}_{1}".format(i, j), lowBound=0)
        for i in set_I
        for j in set_J
    }

    # if x is Binary
    # x_vars = {
    #     (i, j): plp.LpVariable(cat=plp.LpBinary, name="x_{0}_{1}".format(i, j)) for i in set_I for j in set_J
    # }

    # # if x is Integer
    # x_vars = {
    #     (i, j): plp.LpVariable(cat=plp.LpInteger, name="x_{0}_{1}".format(i, j)) for i in set_I for j in set_J
    # }

    # == constraints
    # constraints = {
    #     ni: plp.LpConstraint(
    #         e=1
    #         + discount_factor
    #         * plp.lpSum(p[i, j, ni][0] * x_vars[i, j] for j in set_J for i in set_I),
    #         sense=plp.LpConstraintEQ,
    #         rhs=plp.lpSum(x_vars[ni, j] for j in set_J),
    #         name="constraint_{0}".format(ni),
    #     )
    #     for ni in set_I
    # }
    opt_model += plp.lpSum(x_vars[i, k] * c[i, k] for i in set_I for k in set_J)
    
    for ni in set_I:
        opt_model += 1 + (discount_factor * plp.lpSum(
            p[i, k, ni][0] * x_vars[i, k] for i in set_I for k in set_J
        )) == plp.lpSum(x_vars[ni, k] for k in set_J)

    # objective = plp.lpSum(x_vars[i, j] * c[i, j][0] for i in set_I for j in set_J)

    opt_model.sense = plp.LpMinimize
    # opt_model.setObjective(objective)

    opt_model.solve()

    for variable in opt_model.variables():
        print("{} = {}".format(variable.name, variable.varValue))

    policy = defaultdict(lambda: np.ones(env.action_space.n) / env.action_space.n)
    # loop over each state
    for state in env.P.keys():
        # compute action values
        action_values = state_values[state]
        # get the best action
        # print(action_values)
        best_action = np.argmax(action_values)
        # print(best_action)
        # update policy
        policy[state] = np.eye(len(env.P[state]))[best_action]
        # print(policy[state])
    return policy, state_values
