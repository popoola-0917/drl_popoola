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

# Initialize the alpha value as 1 for both arms:
alpha = np.ones(2)

# Initialize the beta value as 1 for both arms:
beta = np.ones(2)




# Defining the thompson_sampling function.
# As the following code shows, we randomly sample values from the beta
# distributions of both arms and return the arm that has the maximum sampled value:

def thompson_sampling(alpha,beta):
	samples = [np.random.beta(alpha[i]+1,beta[i]+1) for i in range(2)]
	return np.argmax(samples)





# Playing the game and try to find the best arm 
# using the Thompson sampling method.

# For each round:
for i in range(num_rounds):

	# Select the arm based on the Thompson sampling method:
	arm = thompson_sampling(alpha,beta)

	# Pull the arm and store the reward and next state information:
	next_state, reward, done, info = env.step(arm)

	# Increment the count of the arm by 1:
	count[arm] += 1

	# Update the sum of rewards of the arm:
	sum_rewards[arm]+=reward

	# Update the average reward of the arm:
	Q[arm] = sum_rewards[arm]/count[arm]


	# If we win the game, that is, if the reward is equal to 1, then we update the value of
	# alpha to 𝛼 = 𝛼+1, else we update the value of beta to 𝛽 = 𝛽 + 1:
	if reward==1:
		alpha[arm] = alpha[arm] + 1
	else:
		beta[arm] = beta[arm] + 1




	# After all the rounds, we can select the optimal arm 
	# as the one that has the highest average reward:
print('The optimal arm is arm {}'.format(np.argmax(Q)+1))


# The preceding code will print:
# The optimal arm is arm 1
# Thus, we found the optimal arm using the Thompson sampling method.