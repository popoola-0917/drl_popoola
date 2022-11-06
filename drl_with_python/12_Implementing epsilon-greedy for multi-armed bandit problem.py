# Implementing the epsilon-greedy method to find the best arm. 

# import the necessary libraries:
import gym
import gym_bandits
import numpy as np

# Creating the bandit with only two arms:
env = gym.make("BanditTwoArmedHighLowFixed-v0")

# Checking the probability distribution of the arm:
print(env.p_dist)

# The preceding code will print:
# [0.8, 0.2]


# We can observe that with arm 1 we win the game with 80% probability and with
# arm 2 we win the game with 20% probability. Here, the best arm is arm 1, as with
# arm 1 we win the game with 80% probability. Now, let's see how to find this best
# arm using the epsilon-greedy method.

# Initializing the variables.
# Initialize the count for storing the number of times an arm is pulled:
count = np.zeros(2)


# Initialize sum_rewards for storing the sum of rewards of each arm:
sum_rewards = np.zeros(2)

# Initialize Q for storing the average reward of each arm:
Q = np.zeros(2)

# Set the number of rounds (iterations):
num_rounds = 100


# Defining the epsilon_greedy function.

# First, we generate a random number from a uniform distribution. 
# If the random number is less than epsilon, then we pull the random arm; 
# else, we pull the best arm that has the maximum average reward, as shown here:

def epsilon_greedy(epsilon):
	if np.random.uniform(0,1) < epsilon:
		return env.action_space.sample()
else:
		return np.argmax(Q)



# Let's play the game and try to find the best arm 
# using the epsilon-greedy method.

# For each round:
for i in range(num_rounds):

# Select the arm based on the epsilon-greedy method:
arm = epsilon_greedy(epsilon=0.5)

# Pull the arm and store the reward and next state information:
next_state, reward, done, info = env.step(arm)

# Increment the count of the arm by 1:
count[arm] += 1

# Update the sum of rewards of the arm:
sum_rewards[arm]+=reward


# Update the average reward of the arm:
Q[arm] = sum_rewards[arm]/count[arm]

# After all the rounds, we look at the average reward obtained from each of the arms:
print(Q)

# The preceding code will print something like this:
# [0.83783784 0.34615385]


# Now, we can select the optimal arm as the one that has the maximum average
# reward:

# Since arm 1 has a higher average reward than arm 2, our optimal arm will be arm 1:
print('The optimal arm is arm {}'.format(np.argmax(Q)+1))

# The preceding code will print:
# The optimal arm is arm 1
# Thus, we have found the optimal arm using the epsilon-greedy method.