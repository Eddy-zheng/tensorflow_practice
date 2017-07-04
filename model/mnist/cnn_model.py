# coding=utf-8
import tensorflow as tf 
import prettytensor as pt  
from tensorflow.examples.tutorials.mnist import input_data

class Model(object):
	"""docstring for model"""
	def __init__(self, image_size = (28,28,1), lr_rate = 1e-3, ):
		self.height, self.width, self.channel = image_size
		self.height, self.width, self.channel = image_size
		self.x = tf.placeholder("float", shape=[None, 784])
		self._create_network(self.x)


	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.01)

		return tf.Variable(initial)

	def bias_variable(self, shape):
		inital = tf.constant(0.01, shape=shape)

		return tf.Variable(inital)

	def conv2d(self, x, w):
		return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	def _create_network(self, x):
		w_conv1 = self.weight_variable([5,5,1,32])
		b_conv1 = self.bias_variable([32])

		x_image = tf.reshape(x, [-1,28,28,1])

		h_conv1 = tf.nn.relu(self.conv2d(x_image, w_conv1) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		w_conv2 = self.weight_variable([5,5,32,64])
		b_conv2 = self.bias_variable([64])

		h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2) + b_conv2)
		h_pool2 = tf.nn.relu(h_conv2)

		w_fc1 = self.weight_variable([7*7*64, 1024])
		b_fc1 = self.bias_variable([1024])

		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

		keep_prob = tf.placeholder('float')
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		w_fc2 = self.weight_variable([1024, 10])
		b_fc2 = self.bias_variable([10])

		y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

		y_ = tf.placeholder("float", shape=[None, 10])
		cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
		

	def train_model(self, mnist):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		sess.run(tf.inital_all_variables())

		for i in range(100):
			batch = mnist.train.next_batch(50)
			train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
			print "step %d, training accuracy %g"%(i, train_accuracy)

		print "test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


if __name__ == '__main__':
	model = Model()






