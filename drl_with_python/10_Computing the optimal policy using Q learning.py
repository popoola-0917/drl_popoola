# Let's implement Q learning to find the 
# optimal policy in the Frozen Lake environment.

# Import the necessary libraries:
import gym
import numpy as np
import random

# Create the Frozen Lake environment using Gym:
env = gym.make('FrozenLake-v0')

# Let's define the dictionary for storing the Q values of the state-action pairs, 
# and initialize the Q values of all the state-action pairs to 0.0:
Q = {}
for s in range(env.observation_space.n):
	for a in range(env.action_space.n):
		Q[(s,a)] = 0.0



# Let's define the epsilon-greedy policy. We generate a random number from the
# uniform distribution, and if the random number is less than epsilon we select the
# random action, else we select the best action that has the maximum Q value:

def epsilon_greedy(state, epsilon):

	if random.uniform(0,1) < epsilon:
		return env.action_space.sample()
	else:
		return max(list(range(env.action_space.n)), key = lambda x: Q[(state,x)])


# Initialize the discount factor 𝛾 , the learning rate 𝛼 , and the epsilon value:
alpha = 0.85
gamma = 0.90
epsilon = 0.8

# Set the number of episodes and the number of time steps in the episode:
num_episodes = 50000
num_timesteps = 1000



# Compute the policy.
# For each episode:
for i in range(num_episodes):

	# Initialize the state by resetting the environment:
	s = env.reset()

	# For each step in the episode:
	for t in range(num_timesteps):

		# Select the action using the epsilon-greedy policy:
		a = epsilon_greedy(s,epsilon)

		# Perform the selected action and store the next state information:
		s_, r, done, _ = env.step(a)

		# Select the action 𝑎′ that has the maximum Q value in the next state 𝑠′:
		a_ = np.argmax([Q[(s_, a)] for a in range(env.action_space.n)])


		# We can compute the Q value of the state-action pair as:
		Q[(s,a)] += alpha * (r + gamma * Q[(s_,a_)]-Q[(s,a)])

		# Update 𝑠 = 𝑠′ (update the next state 𝑠′ to the current state s):
		s = s_

		# If the current state is the terminal state, then break:
		if done:
			break


