# Import the necessary libraries
import gym
import numpy as np

# Let's create the Frozen Lake environment using Gym
env = gym.make('FrozenLake-v0')



# COMPUTING THE VALUE FUNCTION USING THE POLICY

# Let's define a function called compute_value_function, 
# which takes the policy as a parameter
def compute_value_function(policy):
	# let's define the number of iterations
	num_iterations = 1000

	# Define the threshold value
	threshold = 1e-20
	
	# Set the discount factor 𝛾  value to 1.0
	gamma = 1.0

	# Now, we will initialize the value table 
	# by setting all the state values to zero
	value_table = np.zeros(env.observation_space.n)

	# For every iteration
	for i in range(num_iterations):
		# Update the value table
		updated_value_table = np.copy(value_table)


		# Now, we compute the value function using the given policy. 
		# Thus, for each state, we select the action according to the policy 
		# and then we update the value of the state using the selected action as follows.

		#For each state:
		for s in range(env.observation_space.n):
			# Select the action in the state according to the policy:
			a = policy[s]

			# Compute the value of the state using the selected action,
			value_table[s] = sum([prob * (r + gamma * updated_value_table[s_])
				for prob, s_, r, _ in env.P[s][a]])


		# After computing the value table, 
		# we check whether the difference between the value table 
		# obtained in the current iteration and the previous iteration 
		# is less than or equal to a threshold value. If it is less, then 
		# we break the loop and return the value table as an accurate value 
		# function of the given policy
		if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
			break
	return value_table


 
# EXTRACTING THE POLICY FROM THE VALUE FUNCTION

# This step is exactly the same as how we extracted the policy 
# from the value function in the value iteration method. 
# Thus, similar to what we learned in the value iteration method, 
# we define a function called extract_policy to extract a policy given the value function:
def extract_policy(value_table):
	gamma = 1.0
    policy = np.zeros(env.observation_space.n)     
    for s in range(env.observation_space.n):
    	Q_values = [sum([prob*(r + gamma * value_table[s_])
    		for prob, s_, r, _ in env.P[s][a]])
    			for a in range(env.action_space.n)]
    	policy[s] = np.argmax(np.array(Q_values))
    return policy


# PUTTING IT ALL TOGETHER 

# First, let's define a function called policy_iteration, 
# which takes the environment as a parameter
def policy_iteration(env):

#Set the number of iterations:
num_iterations = 1000


# We learned that in the policy iteration method, 
# we begin by initializing a random policy. 
# So, we will initialize the random policy, 
# which selects the action 0 in all the states
policy = np.zeros(env.observation_space.n)


# For every iteration:
for i in range(num_iterations):
	# Compute the value function using the policy:
	value_function = compute_value_function(policy)

	# Extract the new policy from the computed value function:
	new_policy = extract_policy(value_function)

	# If policy and new_policy are the same, then break the loop:
	if (np.all(policy == new_policy)):
		break

	# Else update the current policy to new_policy
	policy = new_policy
return policy

# Now, let's learn how to perform policy iteration and 
# find the optimal policy in the Frozen Lake environment. 
# So, we just feed the Frozen Lake environment to our policy_iteration 
# function as shown here and get the optimal policy
optimal_policy = policy_iteration(env)


# We can print the optimal policy:
print(optimal_policy)