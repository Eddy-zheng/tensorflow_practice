# coding=utf-8
import tensorflow as tf 
import prettytensor as pt  
from tensorflow.examples.tutorials.mnist import input_data

class model(object):
	"""docstring for model"""
	def __init__(self, image_size = (28,28,1), lr_rate = 1e-3, ):
		self.height, self.width, self.channel = image_size
		self.height, self.width, self.channel = image_size
		self.l2_rate = l2_rate
		self.dropout = dropout
		self.nClasses = nClasses
		self.create_network()

	def create_network(self):
		self.images = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.channel))
		self.sequence_length = tf.placeholder(tf.int32, shape=(None))
		self.labels_indices = tf.placeholder(tf.int64)
		self.labels_values = tf.placeholder(tf.int32)
		self.labels_shape = tf.placeholder(tf.int64)
		self.labels = tf.SparseTensor(indices=self.labels_indices, values=self.labels_values, shape=self.labels_shape)

		self.images_wrap = pt.wrap(self.images)
		with tf.variable_scope('cnn'):
			self.cnn_train = self.cnn(pt.Phase.train)	#size: 25, ?, 13
		with tf.variable_scope('cnn', reuse=True):
			self.cnn_test = self.cnn(pt.Phase.test)

		self.loss_batch = tf.nn.ctc_loss(self.cnn_train, self.labels, self.sequence_length, preprocess_collapse_repeated=False, ctc_merge_repeated=True)
		l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() ]) * self.l2_rate
		self.loss = tf.reduce_mean(self.loss_batch) + l2_loss

		self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.cnn_train, self.sequence_length, merge_repeated=True)
		self.decoded_test, self.log_prob_test = tf.nn.ctc_greedy_decoder(self.cnn_test, self.sequence_length, merge_repeated=True)

	def cnn(self, phase):
			with pt.defaults_scope(activation_fn=tf.nn.relu):
				# batch * 6 * 25 * 64
				last_full = self.images_wrap \
				.conv2d(3, 16).max_pool(2, 2) \
				.conv2d(3, 32).max_pool(2, 2) \
				.conv2d(3, 64)
				# .conv2d(7, 64).max_pool(2, 2) \

			print last_full.get_shape()

			# (25*batch) * (6*64)
			timebatch_feature = tf.reshape(tf.transpose(last_full, [2, 0, 1, 3]), (-1, 12*64))

			# 25 * batch * nClasses
			timebatch_softmax = pt.wrap(timebatch_feature) \
			.fully_connected(64, activation_fn=tf.nn.relu) \
			.fully_connected(self.nClasses, activation_fn=None) \
			.reshape((50, -1, self.nClasses))
			
			print timebatch_softmax.get_shape()

			# return tf.clip_by_value(timebatch_softmax, -1, 1)
			return timebatch_softmax
