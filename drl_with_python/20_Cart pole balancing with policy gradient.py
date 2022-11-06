# This code implement the policy gradient algorithm 
# with reward-to-go for the cart pole balancing task.


# Import the necessary libraries:
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import gym


# Creating the cart pole environment using gym:
env = gym.make('CartPole-v0')

# Get the state shape:
state_shape = env.observation_space.shape[0]

# Get the number of actions:
num_actions = env.action_space.n



# Computing discounted and normalized reward

# Instead of using the rewards directly, we can 
# use the discounted and normalized rewards.



# Set the discount factor, 𝛾:
gamma = 0.95

# Define a function called discount_and_normalize_rewards 
# for computing the discounted and normalized rewards:
def discount_and_normalize_rewards(episode_rewards):


	# Initialize an array for storing the discounted rewards:
	discounted_rewards = np.zeros_like(episode_rewards)

	# Compute the discounted reward:
	reward_to_go = 0.0

	for i in reversed(range(len(episode_rewards))):
		reward_to_go = reward_to_go * gamma + episode_rewards[i]
		discounted_rewards[i] = reward_to_go

	# Normalize the reward:
	discounted_rewards -= np.mean(discounted_rewards)
	discounted_rewards /= np.std(discounted_rewards)

	# Return the reward:
	return discounted_rewards






# Building the policy network

# Define the placeholder for the state:
state_ph = tf.placeholder(tf.float32, [None, state_shape], name="state_ph")

# Define the placeholder for the action:
action_ph = tf.placeholder(tf.int32, [None, num_actions], name="action_ph")


# Define the placeholder for the discounted reward:
discounted_rewards_ph = tf.placeholder(tf.float32, [None,], name="discounted_rewards")


# Define the first layer:
layer1 = tf.layers.dense(state_ph, units=32, activation=tf.nn.relu)

# Define the second layer.
# It is important to note that the number of units in
# layer 2 is set to the number of actions:
layer2 = tf.layers.dense(layer1, units=num_actions)


# Obtain the probability distribution over the action space as an output of the network
# by applying the softmax function to the result of the second layer:
prob_dist = tf.nn.softmax(layer2)



# it is a standard convention to perform minimization rather than maximization.
# So, we can convert the preceding maximization objective into the minimization objective 
# by just adding a negative sign. We can implement this using
# "tf.nn.softmax_cross_entropy_with_logits_v2". 
# Thus, we can define the negative log policy as:

neg_log_policy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = layer2, labels = action_ph)

# Defining the loss:
loss = tf.reduce_mean(neg_log_policy * discounted_rewards_ph)

# Define the train operation for minimizing the loss using the Adam optimizer:
train = tf.train.AdamOptimizer(0.01).minimize(loss)




# Training the network

# Set the number of iterations:
num_iterations = 1000

# Start the TensorFlow session:
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(num_iterations):
		episode_states, episode_actions, episode_rewards = [],[],[]
		done = False
		Return = 0
		state = env.reset()



		while not done:

			state = state.reshape([1,4])
			pi = sess.run(prob_dist, feed_dict={state_ph: state})
			a = np.random.choice(range(pi.shape[1]), p=pi.ravel())
			next_state, reward, done, info = env.step(a)
			env.render()
			Return += reward


			action = np.zeros(num_actions)
			action[a] = 1


			episode_states.append(state)
			episode_actions.append(action)
			episode_rewards.append(reward)


			state=next_state


			discounted_rewards= discount_and_normalize_rewards(episode_rewards)




			feed_dict = {state_ph: np.vstack(np.array(episode_states)), 
						action_ph: np.vstack(np.array(episode_actions)), 
						discounted_rewards_ph: discounted_rewards
						}


			loss_, _ = sess.run([loss, train], feed_dict=feed_dict)


			if i%10==0:


				print("Iteration:{}, Return: {}".format(i,Return))