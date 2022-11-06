# IMPORTING THE REQUIRED LIBRARIES

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity(tf.logging.ERROR)

import matplotlib.pyplot as plt
%matplotlib inline



# LOADING THE DATASET

# Load the dataset, using the following code:
mnist = input_data.read_data_sets("data/mnist", one_hot=True)


# In the preceding code, data/mnist implies the location where we store the MNIST
# dataset, and one_hot=True implies that we are one-hot encoding the labels (0 to 9).


# Let's see what we have in our data by executing the following code:
print("No of images in training set {}".format(mnist.train.images.shape))
print("No of labels in training set {}".format(mnist.train.labels.shape))
print("No of images in test set {}".format(mnist.test.images.shape))
print("No of labels in test set {}".format(mnist.test.labels.shape))


# No of images in training set (55000, 784)
# No of labels in training set (55000, 10)
# No of images in test set (10000, 784)
# No of labels in test set (10000, 10)
# We have 55000 images in the training set, each image is of size 784, and we have 10
# labels, which are actually 0 to 9. Similarly, we have 10000 images in the test set.


# Let plot an input image to see what it looks like:
img1 = mnist.train.images[0].reshape(28,28)
plt.imshow(img1, cmap='Greys')



# DEFINING THE NUMBER OF NEURONS IN EACH LAYER

# We'll build a four-layer neural network with three hidden layers and one output
# layer. As the size of the input image is 784, we set num_input to 784, and since we
# have 10 handwritten digits (0 to 9), we set 10 neurons in the output layer.

# We define the number of neurons in each layer as follows:
#number of neurons in input layer
num_input = 784
#num of neurons in hidden layer 1
num_hidden1 = 512
#num of neurons in hidden layer 2
num_hidden2 = 256
#num of neurons in hidden layer 3
num_hidden_3 = 128
#num of neurons in output layer
num_output = 10




# DEFINING THE PLACEHOLDERS
# As we have learned, we first need to define the placeholders for input and output.
# Values for the placeholders will be fed in at runtime through feed_dict:
with tf.name_scope('input'):
X = tf.placeholder("float", [None, num_input])
with tf.name_scope('output'):
Y = tf.placeholder("float", [None, num_output])


# Since we have a four-layer network, we have four weights and four biases. 
# We initialize our weights by drawing values from the truncated normal 
# distribution with a standard deviation of 0.1.


# Note that the dimensions of the weight matrix should
# be the number of neurons in the previous layer x the number 
# of neurons in the current layer.


# For instance, the dimensions of weight matrix w3 should be the 
# number of neurons in hidden layer 2 x the number of neurons in hidden layer 3.


# We often define all of the weights in a dictionary, as follows:
with tf.name_scope('weights'):
weights = {
'w1': tf.Variable(tf.truncated_normal([num_input, num_hidden1],
stddev=0.1),name='weight_1'),
'w2': tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2],
stddev=0.1),name='weight_2'),
'w3': tf.Variable(tf.truncated_normal([num_hidden2, num_hidden_3],
stddev=0.1),name='weight_3'),
'out': tf.Variable(tf.truncated_normal([num_hidden_3, num_output],
stddev=0.1),name='weight_4'),
}



# The shape of the bias should be the number of neurons in the current layer. 
# For instance, the dimension of the b2 bias is the number of neurons in hidden layer 2.
# We set the bias value as a constant; 0.1 in all of the layers:

with tf.name_scope('biases'):
biases = {
'b1': tf.Variable(tf.constant(0.1, shape=[num_
hidden1]),name='bias_1'),
'b2': tf.Variable(tf.constant(0.1, shape=[num_
hidden2]),name='bias_2'),
'b3': tf.Variable(tf.constant(0.1, shape=[num_
hidden_3]),name='bias_3'),
'out': tf.Variable(tf.constant(0.1, shape=[num_
output]),name='bias_4')
}




# FORWARD PROPAGATION

# Now we'll define the forward propagation operation. 
# We'll use ReLU activations in all layers. 
# In the last layers, we'll apply sigmoid activation

with tf.name_scope('Model'):
with tf.name_scope('layer1'):
layer_1 = tf.nn.relu(tf.add(tf.matmul(X, weights['w1']),
biases['b1']) )
with tf.name_scope('layer2'):
layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']),
biases['b2']))
with tf.name_scope('layer3'):
layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['w3']),
biases['b3']))
with tf.name_scope('output_layer'):
y_hat = tf.nn.sigmoid(tf.matmul(layer_3, weights['out']) +
biases['out'])


# COMPUTING LOSS AND BACKPROPAGATION

# Next, we'll define our loss function. We'll use softmax cross-entropy as our loss
# function. TensorFlow provides the tf.nn.softmax_cross_entropy_with_logits()
# function for computing softmax cross-entropy loss. It takes two parameters as inputs,
# logits and labels:
# • The logits parameter specifies the logits predicted by our network; 
# • The labels parameter specifies the actual labels; 
# We take the mean of the loss function using tf.reduce_mean():

with tf.name_scope('Loss'):
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_
logits(logits=y_hat,labels=Y))


# Now, we need to minimize the loss using backpropagation.
# We don't have to calculate the derivatives of all the weights manually. 
# Instead, we can use TensorFlow's optimizer:
learning_rate = 1e-4
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)



# COMPUTING THE ACCURACY

# We calculate the accuracy of our model as follows:
# • The y_hat parameter denotes the predicted probability for each class of
# our model. Since we have 10 classes, we will have 10 probabilities. If the
# probability is high at position 7, then it means that our network predicts
# the input image as digit 7 with high probability. The tf.argmax() function
# returns the index of the largest value. Thus, tf.argmax(y_hat,1) gives the
# index where the probability is high. Thus, if the probability is high at index 7,
# then it returns 7.

# • The Y parameter denotes the actual labels, and they are the one-hot encoded
# values. That is, the Y parameter consists of zeros everywhere except at the
# position of the actual image, where it has a value of 1. For instance, if the
# input image is 7, then Y has a value of 0 at all indices except at index 7, where
# it has a value of 1. Thus, tf.argmax(Y,1) returns 7 because that is where we
# have a high value, 1.


# Thus, tf.argmax(y_hat,1) gives the predicted digit, and tf.argmax(Y,1) gives us
# the actual digit.
# The tf.equal(x, y) function takes x and y as inputs and returns the truth value of
# (x == y) element-wise. Thus, correct_pred = tf.equal(predicted_digit,actual_
# digit) consists of True where the actual and predicted digits are the same, and
# False where the actual and predicted digits are not the same. We convert the
# Boolean values in correct_pred into float values using TensorFlow's cast operation,
# tf.cast(correct_pred, tf.float32). After converting them into float values, we
# take the average using tf.reduce_mean().
# Thus, tf.reduce_mean(tf.cast(correct_pred, tf.float32)) gives us the average
# correct predictions:

with tf.name_scope('Accuracy'):
	predicted_digit = tf.argmax(y_hat, 1)
	actual_digit = tf.argmax(Y, 1)
	correct_pred = tf.equal(predicted_digit,actual_digit)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# CREATING A SUMMARY
# We can also visualize how the loss and accuracy of our model changes during
# several iterations in TensorBoard. So, we use tf.summary() to get the summary of
# the variable. Since the loss and accuracy are scalar variables, we use tf.summary.
# scalar(), as shown in the following code:

tf.summary.scalar("Accuracy", accuracy)
tf.summary.scalar("Loss", loss)


# Next, we merge all of the summaries we use in our graph, using tf.summary.
# merge_all(). We do this because when we have many summaries, running and
# storing them would become inefficient, so we run them once in our session instead
# of running them multiple times:

merge_summary = tf.summary.merge_all()



# TRAINING THE MODEL

# Firstly, we need to initialize all of the variables:
init = tf.global_variables_initializer()

# Define the batch size, number of iterations, and learning rate, as follows:
learning_rate = 1e-4
num_iterations = 1000
batch_size = 128

# Start the TensorFlow session:
with tf.Session() as sess:

	# Initialize all the variables:
	sess.run(init)

	# Save the event files:
	summary_writer = tf.summary.FileWriter('./graphs', graph=tf.get_default_graph()
	
	# Train the model for a number of iterations:
	for i in range(num_iterations):


		# Get a batch of data according to the batch size:
		batch_x, batch_y = mnist.train.next_batch(batch_size)

		# Train the network:
		sess.run(optimizer, feed_dict={ X: batch_x, Y: batch_y})


		# Print loss and accuracy for every 100th iteration:
		if i % 100 == 0:
			batch_loss, batch_accuracy,summary = sess.run([loss, accuracy, merge_summary], 
				feed_dict={X: batch_x, Y: batch_y})
			
			#store all the summaries
			summary_writer.add_summary(summary, i)

			print('Iteration: {}, Loss: {}, Accuracy: {}'.format(i,batch_loss,batch_accuracy))



# VISUALIZING THE GRAPH USING TENSOR BOARD

# After training, we can visualize our computational graph in TensorBoard.
# If we double-click and expand Model, we can see that we have three hidden layers
# and one output layer:
# Similarly, we can double-click and see every node.


# Note that we also stored a summary of our loss and accuracy variables. 
# We can find them under the SCALARS tab in TensorBoard.

