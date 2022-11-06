# Let's now understand how to implement every-visit MC prediction with the blackjack game step by step:

# Import the necessary libraries:
import gym
import pandas as pd
from collections import defaultdict

Create a blackjack environment:
env = gym.make('Blackjack-v0')


# Defining a policy 
# our policy function takes the state as an input and if the state[0], 
# the sum of our cards, value, is greater than 19, then it will return action 0 (stand), 
# else it will return action 1 (hit):
def policy(state):
	return 0 if state[0] > 19 else 1


# We defined an optimal policy: it makes more sense to perform an action 0 (stand) 
# when our sum value is already greater than 19. That is, when the sum value is greater 
# than 19 we don't have to perform a 1 (hit) action and receive a new card, which may 
# cause us to lose the game or bust. For example, let's generate an initial state by 
# resetting the environment as shown as follows:

#state = env.reset() 
#print(state)

# Suppose the preceding code prints the following: (20, 5, False)

# As we can notice, state[0] = 20; that is, the value of the sum of our cards is 20, 
# so in this case, our policy will return the action 0 (stand) as the following shows:
#print(policy(state))

# The preceding code will print:
#0


# GENERATING AN EPISODE
# Next, we generate an episode using the given policy, 
# so we define a function called generate_episode, which takes 
# the policy as an input and generates the episode using the given policy.
# First, let's set the number of time steps:

num_timesteps = 100

# For a clear understanding, let's look into the function line by line:
def generate_episode(policy):
	# Let's define a list called episode for storing the episode:
	episode = []
	# Initialize the state by resetting the environment:
	state = env.reset()
	# Then for each time step:
	for t in range(num_timesteps):
		# Select the action according to the given policy:
		action = policy(state)
		# Perform the action and store the next state information:
		next_state, reward, done, info = env.step(action)

		# Store the state, action, and reward into our episode list:
		episode.append((state, action, reward))

		# If the next state is a final state then break the loop, 
		# else update the next state to the current state:
		if done:
			break
		state = next_state
	return episode


		# Let's take a look at what the output of our generate_episode function looks like. 
		# Note that we generate an episode using the policy we defined earlier:
print(generate_episode(policy))

# The preceding code will print something like the following: [((10, 2, False), 1, 0), ((20, 2, False), 0, 1.0)]
# As we can observe our output is in the form of [(state, action, reward)]. 
# As shown previously, we have two states in our episode. 
# We performed action 1 (hit) in the state (10, 2, False) and received a 0 reward, 
# and we performed action 0 (stand) in the state (20, 2, False) and received a reward of 1.0.
# Now that we have learned how to generate an episode using the given policy, 
# next, we will look at how to compute the value of the state (value function) 
# using the every-visit MC method.

# COMPUTING THE VALUE FUNCTION
# We learned that in order to predict the value function, 
# we generate several episodes using the given policy and 
# compute the value of the state as an average return across several episodes. 
# Let's see how to implement that. 
# First, we define the total_return and N as a dictionary for storing the total return and 
# the number of times the state is visited across episodes respectively: 
total_return = defaultdict(float) 
N = defaultdict(int)

# Set the number of iterations, that is, the number of episodes, we want to generate:
num_iterations = 500000
# Then, for every iteration:
for i in range(num_iterations):
	# Generate the episode using the given policy; that is, 
	# generate an episode using the policy function we defined earlier:
	episode = generate_episode(policy)

	# Store all the states, actions, and rewards obtained from the episode:
	states, actions, rewards = zip(*episode)

	# Then, for each step in the episode:
	for t, state in enumerate(states):
		# Compute the return R of the state as the sum of rewards,
		R = (sum(rewards[t:]))
		# Update the total_return of the state 
		total_return[state] =  total_return[state] + R
		# Update the number of times the state is visited in the episode
		N[state] =  N[state] + 1

	# After computing the total_return and N we can just convert them into a pandas data frame 
	# for a better understanding. Note that this is just to give a clear understanding of the algorithm; 
	# we don't necessarily have to convert to the pandas data frame, we can also implement this efficiently 
	# just by using the dictionary.

# Convert the total_returns dictionary into a data frame:
total_return = pd.DataFrame(total_return.items(),columns=['state', 'total_return'])

# Convert the counter N dictionary into a data frame:
N = pd.DataFrame(N.items(),columns=['state', 'N'])

# Merge the two data frames on states:
df = pd.merge(total_return, N, on="state")

# Look at the first few rows of the data frame:
df.head(10)

# Next, we can compute the value of the state as the average return:
df['value'] = df['total_return']/df['N']

# Let's look at the first few rows of the data frame:
df.head(10)

# As we can observe, we now have the value of the state, which is just the average of a return 
# of the state across several episodes. Thus, we have successfully predicted the value function of 
# the given policy using the every-visit MC method. 
# Okay, let's check the value of some states and understand how accurately our 
# value function is estimated according to the given policy. Recall that when we started off, 
# to generate episodes, we used the optimal policy, which selects the action 0 (stand) when the sum 
# value is greater than 19 and the action 1 (hit) when the sum value is lower than 19. 


# Let's evaluate the value of the state (21,9,False), as we can observe, 
# the value of the sum of our cards is already 21 and so this is a good state 
# and should have a high value. Let's see what our estimated value of the state is:
df[df['state']==(21,9,False)]['value'].values


# As we can observe, the value of the state is high. Now, let's check the value of the state (5,8,False). 
# As we can notice, the value of the sum of our cards is just 5 and even the one dealer's single card has a 
# high value, 8; in this case, the value of the state should be lower. Let's see what our estimated value of 
# the state is:
df[df['state']==(5,8,False)]['value'].values

# The preceding code will print something like this: array([-1.0])
# As we can notice, the value of the state is lower.
# Thus, we learned how to predict the value function of the given policy using the 
# every-visit MC prediction method. In the next section, we will look at how to compute the value of 
# the state using the first-visit MC method. 