import numpy as np
import pprint
from collections import defaultdict

def prioritized_sweeping(env, discount_factor=1.0, theta=0.00001, alpha=0.1):
	pp = pprint.PrettyPrinter(indent=4)
	N = 10
	actions = range(env.nA)

	state = env.reset()
	state_action_values = {}
	model_transitions = {}
	model_rewards = {}
	previous_states = {}
		
	init_state(state, state_action_values, model_transitions, model_rewards, previous_states, env.nA)

	priority_queue = []
	
	for _ in range(500):
		action = policy(state, actions, state_action_values)
		# we only receive the state and reward from the environment 
		# because this is sequential learning
		(new_state, reward, _, _) = env.step(action)

		if new_state not in model_transitions:
			init_state(new_state, state_action_values, model_transitions, model_rewards, previous_states, env.nA)

		if (state, action) not in previous_states[new_state]:
			previous_states[new_state] += [(state, action)]

		model_transitions[state][action][new_state] = model_transitions[state][action].get(new_state, 0) + 1
		model_rewards[state][action][new_state] = (model_rewards[state][action].get(new_state, 0) * 
												  	(model_transitions[state][action][new_state] - 1) + reward) / \
												   model_transitions[state][action][new_state]
		p = compute_priority(state_action_values, state, action, new_state, reward, discount_factor)
		if p > theta:
			priority_queue += [(state, action, p)]
		for i in range(N):
			if len(priority_queue) == 0:
				break
			(s, a, p) = max(priority_queue, key=lambda x: x[2])
			priority_queue.remove((s, a, p))
			new_s = get_transition(model_transitions, s, a)
			reward = model_rewards[s][a][new_s]
			state_action_values[s][a] = state_action_values[s][a] + \
										alpha * (reward +  \
											 	discount_factor * max(state_action_values[new_s]) - \
											 	state_action_values[s][a])
			
			for (p_state, p_action) in previous_states[s]:
				p_reward = model_rewards[p_state][p_action][s]
				p = compute_priority(state_action_values, p_state, p_action, s, reward, discount_factor)
				if p > theta:
					priority_queue += [(p_state, p_action, p)]
		if env.terminal_states[new_state]:
			state = env.reset()
		else:
			state = new_state

	for state in model_transitions:
		for action in model_transitions[state]:
			sum_values = sum(model_transitions[state][action].values()) 
			for new_state in model_transitions[state][action]:
				model_transitions[state][action][new_state] /= sum_values
	final_policy = defaultdict(lambda: np.ones(env.action_space.n) / env.action_space.n)
	for state in state_action_values:
		final_policy[state] = np.zeros(env.action_space.n)
		final_policy[state][np.argmax(state_action_values[state])] = 1.0
	'''
	print("Model transitions")
	pp.pprint(model_transitions)
	print("Model rewards")
	pp.pprint(model_rewards)
	print("State Action values")
	pp.pprint(state_action_values)
	'''
	return final_policy

def policy(state, actions, state_action_values):
	# Epsilon greedy
	epsilon = 0.2
	if np.random.uniform(low=0.0, high=1.0) < epsilon:
		return np.random.choice(actions)
	return np.argmax(state_action_values[state])

def get_transition(model, state, action):
	probabilities = np.array(list(model[state][action].values())) / sum(model[state][action].values())
	state = np.random.choice(np.array(list(model[state][action].keys())), p=probabilities)
	return state

def compute_priority(state_action_values, state, action, new_state, reward, discount_factor):
	p = abs(reward + 
			discount_factor * max(state_action_values[new_state]) - 
			state_action_values[state][action])
	return p

def init_state(state, state_action_values, model_transitions, model_rewards, previous_states, nA):
	state_action_values[state] = [0] * nA
	model_transitions[state] = {i:{} for i in range(nA)}
	model_rewards[state] = {i:{} for i in range(nA)}
	previous_states[state] = []
	