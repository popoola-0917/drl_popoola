# Instead of creating the bandit environment from scratch, 
# we will use the open-source version
# of the bandit environment provided by Jesse Cooper.

# Cloning the Gym bandits repository:
git clone https://github.com/JKCooper2/gym-bandits

# Installing the bandit environment using pip:
cd gym-bandits
pip install -e 


# import gym_bandits and the gym library
# gym_bandits provides several versions of the bandit environment.
import gym_bandits
import gym


# 1. Creating a bandit in the Gym


# Creating a simple 2-armed bandit whose environment ID 
# is BanditTwoArmedHighLowFixed-v0:
env = gym.make("BanditTwoArmedHighLowFixed-v0")


# Since we created a 2-armed bandit, 
# our action space will be 2 (as there are two arms), as shown here:
print(env.action_space.n)

# The preceding code will print:
# 2


# Checking the probability distribution of the arm with:
print(env.p_dist)

# The preceding code will print:
# [0.8, 0.2]

# It indicates that, with arm 1, we win the game 80% of the 
# time and with arm 2, we win the game 20% of the time. 
# Our goal is to find out whether pulling arm 1 or arm 2
# makes us win the game most of the time.




