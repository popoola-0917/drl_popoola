# TensorFlow is an open source software library from Google, which is extensively
# used for numerical computation. It is one of the most used libraries for building
# deep learning models.

# You can install TensorFlow easily through pip 
# by just typing the following command in your terminal:
# Note that stable baseline only support tensorflow versions < 2.0.0
pip install tensorflow==1.13.1


# We can check the successful installation of TensorFlow 
# by running the following simple Hello TensorFlow! program:
import tensorflow as tf
hello = tf.constant("Hello TensorFlow!")
sess = tf.Session()
print(sess.run(hello))

# The preceding program should print Hello TensorFlow!.