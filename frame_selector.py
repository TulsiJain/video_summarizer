import tensorflow as tf

n_classes = 1
rnn_size = 1024
rnn_layer = 2

def fulconn_layer(input_data, activation_func=None):
    bacth_size = int(input_data.get_shape()[0])
    input_dim = int(input_data.get_shape()[2])

    # ef  = tf.sqrt(input_dim1)
    with tf.variable_scope("bi_directional_lstm"):
        W = tf.get_variable("frame_sel_weights", initializer= tf.random_normal([input_dim, n_classes]))
        b = tf.get_variable("frame_sel_biased", initializer= tf.random_normal([n_classes]))

    expaned = tf.expand_dims(W, 0)
    W = tf.tile(expaned, [bacth_size, 1, 1])

    if activation_func:
        return  tf.reshape(activation_func(tf.matmul(input_data, W)) + b, [bacth_size,-1])
    else:
        return  tf.reshape(tf.matmul(input_data, W) + b, [bacth_size,-1])

def frame_selector_model(data):

    with tf.variable_scope("bi_directional_lstm"):


        stacked_rnn = []
        for iiLyr in range(rnn_layer):
            stacked_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size, state_is_tuple=True))

        stacked_rnn1 = []
        for iiLyr1 in range(rnn_layer):
            stacked_rnn1.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size, state_is_tuple=True))

        lstm_multi_fw_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)
        lstm_multi_bw_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn1)

        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_multi_fw_cell,
            cell_bw=lstm_multi_bw_cell,
            inputs=data,
            dtype=tf.float32)
        # As we have Bi-LSTM, we have two output, which are not connected. So merge them
        outputs = tf.concat(outputs, 2)

    # # As we want do classification, we only need the last output from LSTM.
    # last_output = outputs[:,0,:]
    # # Create the final classification layer
        score = fulconn_layer(outputs)
    # score = tf.reshape(score, [input_dim])

    score = tf.to_float(score)

    score = tf.nn.l2_normalize(score, 0, epsilon=1e-12, name=None)
    # score = tf.div(
    #    tf.subtract(
    #       score,
    #       tf.reduce_min(score)
    #    ),
    #    tf.subtract(
    #       tf.reduce_max(score),
    #       tf.reduce_min(score)
    #    )
    # )
    return score



def train_frame_selector_model(data):
    # batch_size = int(data.get_shape()[0])
    output = frame_selector_model(data)
    # score = tf.reduce_mean(output)
    return output
	# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
	# optimizer = tf.train.AdamOptimizer().minimize(cost)

# 	with tf.Session() as sess:
#     # OLD:
#     #sess.run(tf.initialize_all_variables())
#     # NEW:
# 	    sess.run(tf.global_variables_initializer())

# 	    for epoch in range(hm_epochs):
# 	        epoch_loss = 0
# 	        for _ in range(int(mnist.train.num_examples/batch_size)):
# 	            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
# 	            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
# 	            epoch_loss += c

# 	        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

# 	    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

# 	    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
# 	    print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

# train_neural_network(x)