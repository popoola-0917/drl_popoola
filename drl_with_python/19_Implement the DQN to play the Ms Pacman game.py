# This code play the Ms Pacman game using DQN algorithm


# Importing the necessary libraries
import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import Adam


# Create the Ms Pacman game environment using open-ai Gym:
env = gym.make("MsPacman-v0")


# Setting the state size:
state_size = (88, 80, 1)

# Getting the number of actions:
action_size = env.action_space.n




# PREPROCESSING THE GAME SCREEN
# We learned that we feed the game state (an image of the game screen) as input to
# the DQN, which is a CNN, and it outputs the Q values of all the actions in the state.
# However, directly feeding the raw game screen image is not efficient, since the raw
# game screen size will be 210 x 160 x 3. This will be computationally expensive.
# To avoid this, we preprocess the game screen and then feed the preprocessed game
# screen to the DQN. First, we crop and resize the game screen image, convert the
# image to grayscale, normalize it, and then reshape the image to 88 x 80 x 1. Next, we
# feed this preprocessed game screen image as input to the CNN, which returns the Q
#  values.



color = np.array([210, 164, 74]).mean()

# Define a function called preprocess_state, which takes the game state
# (image of the game screen) as an input and returns the preprocessed game state:
def preprocess_state(state):

# Crop and resize the image:
image = state[1:176:2, ::2]

# Convert the image to grayscale:
image = image.mean(axis=2)

# Improve the image contrast:
image[image==color] = 0

# Normalize the image:
image = (image - 128) / 128 - 1

# Reshape the image:
image = np.expand_dims(image.reshape(88, 80, 1), axis=0)

# Return the image
return image



# DEFINING THE DQN CLASS
# Let's define the class called DQN where we will implement the DQN algorithm. 
class DQN:
	# Define the init method
	def __init__(self, state_size, action_size):

		# Define the state size:
		self.state_size = state_size

		# Define the action size:
		self.action_size = action_size

		# Define the replay buffer:
		self.replay_buffer = deque(maxlen=5000)

		# Define the discount factor:
		self.gamma = 0.9

		# Define the epsilon value:
		self.epsilon = 0.8

		# Define the update rate at which we want to update the target network:
		self.update_rate = 1000

		# Define the main network:
		self.main_network = self.build_network()

		# Define the target network:
		self.target_network = self.build_network()

		# Copy the weights of the main network to the target network:
		self.target_network.set_weights(self.main_network.get_weights())





# BUILDING THE DQN
# Now, let's build the DQN. We have learned that to play Atari games, we use a CNN
# as the DQN, which takes the image of the game screen as an input and returns the Q
# values. We define the DQN with three convolutional layers. The convolutional layers
# extract the features from the image and output the feature maps, and then we flatten
# the feature map obtained by the convolutional layers and feed the flattened feature
# maps to the feedforward network (the fully connected layer), which returns the Q
# values:

def build_network(self):

	# Define the first convolutional layer:
	model = Sequential()
	model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_size))
	model.add(Activation('relu'))

	# Define the second convolutional layer:
	model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
	model.add(Activation('relu'))

	# Define the third convolutional layer:
	model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
	model.add(Activation('relu'))

	# Flatten the feature maps obtained as a result of the third convolutional layer:
	model.add(Flatten())


	# Feed the flattened maps to the fully connected layer:
	model.add(Dense(512, activation='relu'))
	model.add(Dense(self.action_size, activation='linear'))

	# Compile the model with loss as MSE:
	model.compile(loss='mse', optimizer=Adam())


	# Return the model:
	return model



# Storing the transition
# We have learned that we train the DQN by randomly sampling a minibatch of
# transitions from the replay buffer. So, we define a function called store_transition,
# which stores the transition information in the replay buffer:

def store_transistion(self, state, action,reward, next_state, done):
	self.replay_buffer.append((state, action,reward, next_state, done))


# Defining the epsilon-greedy policy
# We learned that in DQN, to take care of exploration-exploitation trade-off, we
# select action using the epsilon-greedy policy. So, now we define the function called
# epsilon_greedy for selecting an action using the epsilon-greedy policy:
def epsilon_greedy(self, state):
	if random.uniform(0,1) < self.epsilon:
		return np.random.randint(self.action_size)
Q_values = self.main_network.predict(state)
return np.argmax(Q_values[0])




# Define the training



# define a function called train for the training network:
def train(self, batch_size):

	# Sample a minibatch of transitions from the replay buffer:
	minibatch = random.sample(self.replay_buffer, batch_size)

	# Compute the target value using the target network:
	for state, action, reward, next_state, done in minibatch:
		if not done:
			target_Q = (reward + self.gamma * np.amax(self.target_network.predict(next_state)))

		else:
			target_Q = reward


		# Compute the predicted value using the main network 
		# and store the predicted value in the Q_values:
		Q_values = self.main_network.predict(state)


		# Update the target value:
		Q_values[0][action] = target_Q

		# Train the main network:
		self.main_network.fit(state, Q_values, epochs=1, verbose=0)



# Updating the target network

# Define the function called update_target_network for updating the target
# network weights by copying from the main network:
def update_target_network(self):
	self.target_network.set_weights(self.main_network.get_weights())



# Training the DQN

# Let's train the network by setting the number of episodes
# we want to train the network for:

num_episodes = 500

# Define the number of time steps:
num_timesteps = 20000

# Define the batch size:
batch_size = 8

# Set the number of past game screens we want to consider:
num_screens = 4

# Instantiate the DQN class:
dqn = DQN(state_size, action_size)

# Set done to False:
done = False

# Initialize the time_step:
time_step = 0

# For each episode:
for i in range(num_episodes):

	# Set Return to 0:
	Return = 0

	# Preprocess the game screen:
	state = preprocess_state(env.reset())

	# For each step in the episode:
	for t in range(num_timesteps):

		# Render the environment:
		# env.render()

		# Update the time step:
		time_step += 1


		# Update the target network:
		if time_step % dqn.update_rate == 0:
			dqn.update_target_network()


		# Select the action:
		action = dqn.epsilon_greedy(state)

		# Perform the selected action:
		next_state, reward, done, _ = env.step(action)

		# Preprocess the next state:
		next_state = preprocess_state(next_state)

		# Store the transition information:
		dqn.store_transistion(state, action, reward, next_state, done)

		# Update the current state to the next state:
		state = next_state

		# Update the return value:
		Return += reward

		# If the episode is done, then print the return:
		if done:
			print('Episode: ',i, ',' 'Return', Return)
			break

		# If the number of transitions in the replay buffer 
		# is greater than the batch size, then train the network:
		if len(dqn.replay_buffer) > batch_size:
			dqn.train(batch_size)

