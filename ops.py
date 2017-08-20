import tensorflow as tf

def dense(x, batch_size, input_dim, output_dim, name, reuse = False):
	'''Fully-connected Layer'''
	with tf.variable_scope(name, reuse = reuse):
		w = tf.get_variable('weights', [input_dim, output_dim], initializer = tf.truncated_normal_initializer(stddev = 0.1))
		b = tf.get_variable('biased', [output_dim], initializer = tf.constant_initializer(0.1))

		# x = tf.reshape(x, [-1, input_dim])
		# y = tf.matmul(x, w) + b

		expaned = tf.expand_dims(w, 0)
		# print(expaned.get_shape())
		w = tf.tile(expaned, [batch_size, 1, 1])

	return tf.matmul(x, w) + b


def selctedFrames(data, indicestobeSelected):
    batch_size_1 = tf.shape(data)[0]
    rows_per_batch = tf.shape(data)[1]
    indices_per_batch = tf.shape(indicestobeSelected)[1]

    # Offset to add to each row in indices. We use `tf.expand_dims()` to make
    # this broadcast appropriately.
    offset = tf.expand_dims(tf.range(0, batch_size_1) * rows_per_batch, 1)

    # Convert indices and logits into appropriate form for `tf.gather()`.
    flattened_indices = tf.reshape(indicestobeSelected + offset, [-1])
    flattened_logits = tf.reshape(data, tf.concat(0, [[-1], tf.shape(data)[2:]]))

    selected_rows = tf.gather(flattened_logits, flattened_indices)

    result = tf.reshape(selected_rows,
                        tf.concat(0, [tf.stack([batch_size_1, indices_per_batch]),
                                      tf.shape(data)[2:]]))

    return result