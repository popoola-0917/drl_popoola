#In this project we will initialize a random policy and 
# predict the value function (state values) of the Frozen 
# Lake environment using the random policy.


# First, let's import the necessary libraries:
import gym
import pandas as pd


# Now, we create the Frozen Lake environment using Gym:
env = gym.make('FrozenLake-v0')

# Define the random policy, which returns the random action 
# by sampling from the action space:
def random_policy():
return env.action_space.sample()


# Let's define the dictionary for storing the value of states, 
# and we initialize the value of all the states to 0.0:
V = {}

for s in range(env.observation_space.n):
	V[s]=0.0

# Initialize the discount factor 𝛾𝛾 and the learning rate 𝛼 :
alpha = 0.85
gamma = 0.90

# Set the number of episodes and the number of time steps in each episode:
num_episodes = 50000
num_timesteps = 1000

# Compute the values of the states
# Now, let's compute the value function (state values) using the given random policy.

# For each episode:
for i in range(num_episodes):


# Initialize the state by resetting the environment:
s = env.reset()

# For every step in the episode:
for t in range(num_timesteps):

	# Select an action according to random policy:
	a = random_policy()

	# Perform the selected action and store the next state information:
	s_, r, done, _ = env.step(a)

	# Compute the value of the state as:
	V[s] += alpha * (r + gamma * V[s_]-V[s])

	# Update the next state to the current state 𝑠=𝑠′:
	s = s_

	# If the current state is the terminal state, then break:
	if done:
		break


	# After all the iterations, we will have values of all 
	# the states according to the given random policy.



# Now, let's evaluate our value function (state values). 
# First, let's convert our value dictionary to a pandas data frame for more clarity:

df = pd.DataFrame(list(V.items()), columns=['state', 'value'])


# Now, Let's check the value of the states:
df