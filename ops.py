import tensorflow as tf

def dense(x, batch_size, input_dim, output_dim, name, reuse = False):
	'''Fully-connected Layer'''
	with tf.variable_scope(name, reuse = reuse):
		w = tf.get_variable('weights',  shape=[input_dim, output_dim], initializer = tf.contrib.layers.xavier_initializer())
		b = tf.get_variable('biased', initializer = tf.zeros([output_dim], dtype=tf.float32))

		expaned = tf.expand_dims(w, 0)
		w = tf.tile(expaned, [batch_size, 1, 1])

	return tf.matmul(x, w) + b