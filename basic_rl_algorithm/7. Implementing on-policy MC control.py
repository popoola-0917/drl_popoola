# Now, let's learn how to implement the MC control method with the epsilon-greedy
# policy to play the blackjack game; that is, we will see how can we use the MC control
# method to find the optimal policy in the blackjack game.

# First, let's import the necessary libraries:
import gym
import pandas as pd
import random
from collections import defaultdict

# Create a blackjack environment:
env = gym.make('Blackjack-v0')

# Initialize the dictionary for storing the Q values:
Q = defaultdict(float)

# Initialize the dictionary for storing the total return of the state-action pair:
total_return = defaultdict(float)

# Initialize the dictionary for storing the count of the number of 
# times a state-action pair is visited:
N = defaultdict(int)

# DEFIINE THE EPSILON-GREEDY POLICY 
# We learned that we select actions based on the epsilon-greedy policy, 
# so we define a function called epsilon_greedy_policy, which takes the state and Q value 
# as an input and returns the action to be performed in the given state:

def epsilon_greedy_policy(state,Q):
	# Set the epsilon value to 0.5:
	epsilon = 0.5
	# Sample a random value from the uniform distribution; if the sampled value is less than 
	# epsilon then we select a random action, else we select the best action that has the 
	# maximum Q value as shown:
	if random.uniform(0,1) < epsilon:
		return env.action_space.sample()     
	else:
		return max(list(range(env.action_space.n)), key = lambda x: Q[(state,x)]) 


# GENERATING AN EPISODE
# Now, let's generate an episode using the epsilon-greedy policy. 
# We define a function called generate_episode, which takes the Q value 
# as an input and returns the episode. 

# First, let's set the number of time steps:
num_timesteps = 100

# let's define the function:
def generate_episode(Q):
	# Initialize a list for storing the episode:
	episode = []
	# Initialize the state using the reset function:
	state = env.reset()
	# Then for each time step:
	for t in range(num_timesteps):
		# Select the action according to the epsilon-greedy policy:
		action = epsilon_greedy_policy(state,Q)
		# Perform the selected action and store the next state information:
		next_state, reward, done, info = env.step(action)
		# Store the state, action, and reward in the episode list:
		episode.append((state, action, reward))
		# If the next state is the final state then break the loop, 
		# else update the next state to the current state:
		if done:
			break
		state = next_state
	return episode

# COMPUTING THE OPTIMAL POLICY
# Now, let's learn how to compute the optimal policy. 
# First, let's set the number of iterations, that is, 
# the number of episodes, we want to generate:
num_iterations = 500000

# For each iteration:
for i in range(num_iterations):

# We learned that in the on-policy control method, we will not be given any policy
# as an input. So, we initialize a random policy in the first iteration and improve the
# policy iteratively by computing the Q value. Since we extract the policy from the Q
# function, we don't have to explicitly define the policy. As the Q value improves, the
# policy also improves implicitly. That is, in the first iteration, we generate the episode
# by extracting the policy (epsilon-greedy) from the initialized Q function. Over a
# series of iterations, we will find the optimal Q function, and hence we also find the
# optimal policy.

# So, here we pass our initialized Q function to generate an episode:
	episode = generate_episode(Q)

	# Get all the state-action pairs in the episode:
	all_state_action_pairs = [(s, a) for (s,a,r) in episode]

	# Store all the rewards obtained in the episode in the rewards list:
	rewards = [r for (s,a,r) in episode]

	# For each step in the episode:
	for t, (state, action,_) in enumerate(episode):
	#If the state-action pair is occurring for the first time in the episode:
		if not (state, action) in all_state_action_pairs[0:t]:
			# Compute the return R of the state-action pair as the sum of rewards, R(st, at) = sum(rewards[t:]):
			R = sum(rewards[t:])
			# Update the total return of the state-action pair as total_return(st, at) = total_return(st,at) + R(st, at):
			total_return[(state,action)] = total_return[(state,action)] + R
			# Update the number of times the state-action pair is visited as N(st, at) = N(st, at) + 1:
			N[(state, action)] += 1
			# Compute the Q value by just taking the average, that is,
			Q[(state,action)] = total_return[(state, action)] / N[(state, action)]

# Thus on every iteration, the Q value improves and so does the policy.
# After all the iterations, we can have a look at the Q value of each state-action pair in
# the pandas data frame for more clarity. First, let's convert the Q value dictionary into
# a pandas data frame:
df = pd.DataFrame(Q.items(),columns=['state_action pair','value'])

# Let's look at the first few rows of the data frame:
df.head(11)

# As we can observe, we have the Q values for all the state-action pairs. Now we can
# extract the policy by selecting the action that has the maximum Q value in each state.
# For instance, say we are in the state (21,8, True). Now, should we perform action 0
# (stand) or action 1 (hit)? It makes more sense to perform action 0 (stand) here, since
# the value of the sum of our cards is already 21, and if we perform action 1 (hit) our
# game will lead to a bust.
# Note that due to stochasticity, you might get different results than those shown here.

# Let's look at the Q values of all the actions in this state, (21,8, True):
df[124:126]

# As we can observe, we have a maximum Q value for action 0 (stand) compared
# to action 1 (hit). So, we perform action 0 in the state (21,8, True). Similarly, in
# this way, we can extract the policy by selecting the action in each state that has the
# maximum Q value.