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



# Now, we define the UCB function, which returns the best arm 
# as the one that has the highest UCB:
def UCB(i):

	# Initialize the numpy array for storing the UCB of all the arms:
	ucb = np.zeros(2)

	# Before computing the UCB, we explore all the arms at least once, so for the first 2
	# rounds, we directly select the arm corresponding to the round number:
	if i < 2:
		return i

	# If the round is greater than 2, then we compute the UCB of all the arms
	# and return the arm that has the highest UCB:
	else:
		for arm in range(2):
			ucb[arm] = Q[arm] + np.sqrt((2*np.log(sum(count))) /count[arm])
		return (np.argmax(ucb))


# Let's play the game and try to find the best arm using the UCB method.

# For each round:
for i in range(num_rounds):
	
	# Select the arm based on the UCB method:
	arm = UCB(i)

	# Pull the arm and store the reward and next state information:
	next_state, reward, done, info = env.step(arm)

	# Increment the count of the arm by 1:
	count[arm] += 1

	# Update the sum of rewards of the arm:
	sum_rewards[arm]+=reward

	# Update the average reward of the arm:
	Q[arm] = sum_rewards[arm]/count[arm]


# After all the rounds, we can select the optimal arm 
# as the one that has the maximum average reward:
print('The optimal arm is arm {}'.format(np.argmax(Q)+1))

# The preceding code will print something like:
# The optimal arm is arm 1
# Thus, we found the optimal arm using the UCB method.

