# Import the necessary libraries
import gym
import numpy as np

# Create the Frozen Lake environment using Gym
env = gym.make('FrozenLake-v0')

# Let's take a look at the Frozen Lake environment using the render function
env.render()


# FROM HERE WE WILL START COMPUTING THE OPTIMAL VALUE FUNCTION
# Define the value_iteration function, which takes the environment as a parameter
def value_iteration(env):
	# Set the number of iterations
	num_iterations = 1000

	# Set the threshold number for checking the convergence of the value function
	threshold = 1e-20

	# Set the discount factor 𝛾  to 1
	gamma = 1.0

	# We will initialize the value table by setting the value of all states to zero
	value_table = np.zeros(env.observation_space.n)

	# For every iteration
	for i in range(num_iterations):
		# Update the value table
		updated_value_table = np.copy(value_table)

		# for each state, we compute the Q values of all the actions 
		# in the state and then we update the value of the state as 
		# the one that has the maximum Q value
		for s in range(env.observation_space.n):
			# Compute the Q value of all the actions
			Q_values = [sum([prob*(r + gamma * updated_value_table[s_])
				for prob, s_, r, _ in env.P[s][a]])
					for a in range(env.action_space.n)]

			# Update the value of the state as a maximum Q value
			value_table[s] = max(Q_values)

		# After computing the value table, we check whether
		# the difference between the value table obtained in 
		# the current iteration and the previous iteration 
		# is less than or equal to a threshold value.
		# If the difference is less than the threshold, 
		# then we break the loop and return the value table 
		# as our optimal value function as the following code shows:
		if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
			break
	return value_table



# FROM HERE WE WILL EXTRACT THE OPTIMAL POLICY FROM 
# THE COMPUTED OPTIMAL VALUE FUNCTION

# We define a function called extract_policy 
# which takes value_table as a parameter
def extract_policy(value_table):
	# Set the discount factor 𝛾  to 1
	gamma = 1.0
	# Let initialize the policy with zeros, 
	# that is, we set the actions for all the states to be zero:
	policy = np.zeros(env.observation_space.n)

	# For each state, we compute the Q values for all the actions 
	# in the state and then we extract the policy by selecting 
	# the action that has the maximum Q value.
	for s in range(env.observation_space.n):
		# Compute the Q value of all the actions in the state, 𝑄(𝑠,a)
		Q_values = [sum([prob*(r + gamma * value_table[s_])
			for prob, s_, r, _ in env.P[s][a]])
				for a in range(env.action_space.n)]

		# Extract the policy by selecting the action that has the maximum Q value,
		policy[s] = np.argmax(np.array(Q_values))
	return policy


#	We learned that in the Frozen Lake environment, 
#	our goal is to find the optimal policy that selects the correct action 
#	in each state so that we can reach state G from state A without visiting the hole states.


# Firstly, we compute the optimal value function 
# using our value_iteration function by passing our 
# Frozen Lake environment as the parameter: 
optimal_value_function = value_iteration(env)

# Next, we extract the optimal policy from the optimal value 
# function using our extract_policy function:
optimal_policy = extract_policy(optimal_value_function)

# We can print the obtained optimal policy:
print(optimal_policy)

# The preceding code will print the following. 
# As we can observe, our optimal policy tells us 
# to perform the correct action in each state: