# Importing the necessary libraries:
import gym
import gym_bandits
import numpy as np


# Creating the bandit with only two arms:
env = gym.make("BanditTwoArmedHighLowFixed-v0")

# Initializing the variables.
# Initialize the count for storing the number of times an arm is pulled:
count = np.zeros(2)


# Initialize sum_rewards for storing the sum of rewards of each arm:
sum_rewards = np.zeros(2)

# Initialize Q for storing the average reward of each arm:
Q = np.zeros(2)

# Set the number of rounds (iterations):
num_rounds = 100




# Now, we define the softmax function with the temperature T:
def softmax(T):
	# Compute the probability of each arm:
	denom = sum([np.exp(i/T) for i in Q])
	probs = [np.exp(i/T)/denom for i in Q]
	
	# Select the arm based on the computed probability distribution of arms:
	arm = np.random.choice(env.action_space.n, p=probs)
	return arm







# Let's play the game and try to find the best arm using 
# the softmax exploration method.

# Let's begin by setting the temperature T to a high number, say, 50:
T = 50

# For each round:
for i in range(num_rounds):

	# Select the arm based on the softmax exploration method:
	arm = softmax(T)

	# Pull the arm and store the reward and next state information:
	next_state, reward, done, info = env.step(arm)

	# Increment the count of the arm by 1:
	count[arm] += 1

	# Update the sum of rewards of the arm:
	sum_rewards[arm]+=reward

	# Update the average reward of the arm:
	Q[arm] = sum_rewards[arm]/count[arm]

	# Reduce the temperature T:
	T = T*0.99




# After all the episodes, we check the Q value, which is the average reward of all the arms:
print(Q)

# The preceding code will print something like this:
# [0.77700348 0.1971831 ]

# As we can see, arm 1 has a higher average reward than arm 2, 
# so we select arm 1 as the optimal arm:
print('The optimal arm is arm {}'.format(np.argmax(Q)+1))

# The preceding code prints:
# The optimal arm is arm 1
# Thus, we have found the optimal arm using the softmax exploration method.



