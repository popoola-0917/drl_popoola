# The code for first-visit MC is the same as what we have seen in 
# every-visit MC except here, we compute the return only for its first 
# time of occurrence as shown in the following highlighted code:
for i in range(num_iterations):
	episode = generate_episode(env,policy)
	states, actions, rewards = zip(*episode)

	for t, state in enumerate(states):
		if state not in states[0:t]:
			R = (sum(rewards[t:]))
			total_return[state] = total_return[state] + R 
			N[state] = N[state] + 1

# Thus, we learned how to predict the value function of the given policy 
# using the first-visit and every-visit MC methods. 