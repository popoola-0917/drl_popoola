# Import the Gym library:
import gym


# The environment id of blackjack is Blackjack-v0. 
# So, we can create the blackjack game using the make function as shown as follows:
env = gym.make('Blackjack-v0')

# Now, let's look at the state of the blackjack environment; 
# we can just reset our environment and look at the initial state:
print(env.reset())


# Note that every time we run the preceding code, 
# we might get a different result, as the initial state is randomly initialized. 
# The preceding code will print something like this:(15, 9, True)

# As we can observe, our state is represented as a tuple, 
# but what does this mean? We learned that in the blackjack game, 
# we will be given two cards and we also get to see one of the dealer's cards. 
# Thus, 15 implies that the value of the sum of our cards, 9 implies the face value of one of the dealer's cards, True implies that we have a usable ace, and it will be False if we don't have a usable ace.
# Thus, in the blackjack environment the state is represented as a tuple consisting of three values:
# 1.	The value of the sum of our cards
# 2.	The face value of one of the dealer's card
# 3.	Boolean value—True if we have a useable ace and False if we don't have a useable ace


# Let's look at the action space of our blackjack environment:
print(env.action_space)

# The preceding code will print: Discrete(2)

# As we can observe, it implies that we have two actions in our action space, which are 0 and 1:
# •	The action stand is represented by 0
# •	The action hit is represented by 1

# The reward will be assigned as follows:
# •	+1.0 reward if we win the game
# •	-1.0 reward if we lose the game
# •	0 reward if the game is a draw