import numpy as np


def prioritized_sweeping(env, discount_factor=1.0, theta=0.00001):
	N = 10
	actions = range(env.nA)

	state = env.reset()

	state_action_values = {}
	model_transitions = {}
	model_rewards = {}
	previous_states = {}
		
	init_state(state, state_action_values, model_transitions, model_rewards, previous_states, env.nA)

	priority_queue = []
	
	for _ in range(100):
		
		action = policy(state, actions, state_action_values)
		# we only receive the state and reward from the environment 
		# because this is sequential learning
		(new_state, reward, _, _) = env.step(action)

		if new_state not in model_transitions:
			init_state(state, state_action_values, model_transitions, model_rewards, previous_states, env.nA)

		if (state, action) not in previous_states[new_state]:
			previous_states[new_state] += [(state, action)]

		model_transitions[state][action][state] = model_transitions[state][action].get(state, 0) + 1
		model_rewards[state][action][state] = model_rewards[state][action].get(state, 0) + reward
		
		p = compute_priority(state_action_values, state, action, new_state, reward, discount_factor)
		if p > theta:
			priority_queue += [(state, action, p)]
		for i in range(N):
			if len(priority_queue) == 0:
				break
			(state, action, p) = max(priority_queue, key=lambda x: x[2])
			priority_queue.remove((state, action, p))
			new_state = get_transition(model_transitions, state, action)
			reward = model_rewards[state][action][new_state]
			state_action_values[state][action] = state_action_values[state].get(action, 0) + \
												 alpha * (reward +  \
												 	discount_factor * max(state_action_values[new_state]) - \
												 	state_action_values[state][action])
			for (p_state, p_action) in previous_states[state]:
				p_reward = model_rewards[p_state][p_action][state]
				p = compute_priority(state_action_values, p_state, p_action, state, reward, discount_factor)
				if p > theta:
					priority_queue += [(p_state, p_action, p)]


def policy(state, actions, state_action_values):
	return np.random.choice(actions)

def get_transition(model, state, action):
	probabilities = np.array(list(model[state][action].values())) / sum(model[state][action].values())
	print(probabilities)
	state = np.random.choice(np.array(list(model[state][action].keys())), p=model[state][state])
	print(state)
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
	