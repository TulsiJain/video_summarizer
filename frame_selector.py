import tensorflow as tf

# we need to output single value, score
n_classes = 1

# number of units in LSTM
rnn_size = 1024

# number of hidden layer in RNN
rnn_layer = 2

def fulconn_layer(input_data, activation_func=None):
    bacth_size = int(input_data.get_shape()[0])
    input_dim = int(input_data.get_shape()[2])

    # ef  = tf.sqrt(input_dim1)
    W = tf.get_variable("frame_sel_weights", shape=[input_dim, n_classes],initializer= tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("frame_sel_biased", initializer= tf.zeros([n_classes]))

    expaned = tf.expand_dims(W, 0)
    W = tf.tile(expaned, [bacth_size, 1, 1])

    if activation_func:
        return  W,b, tf.reshape(activation_func(tf.matmul(input_data, W)) + b, [bacth_size,-1])
    else:
        return  W,b, tf.reshape(tf.matmul(input_data, W) + b, [bacth_size,-1])

def frame_selector_model(data, sequence_length):

    with tf.variable_scope("bi_directional_lstm"):
        stacked_rnn = []
        for iiLyr in range(rnn_layer):
            stacked_rnn.append(tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size, state_is_tuple=True))

        stacked_rnn1 = []
        for iiLyr1 in range(rnn_layer):
            stacked_rnn1.append(tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size, state_is_tuple=True))

        lstm_multi_fw_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)
        lstm_multi_bw_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn1)

        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_multi_fw_cell,
            cell_bw=lstm_multi_bw_cell,
            sequence_length=sequence_length,
            inputs=data,
            dtype=tf.float32)
        # As we have Bi-LSTM, we have two output, which are not connected. So merge them
        outputs = tf.concat(outputs, 2)

    # # As we want do classification, we only need the last output from LSTM.
    # last_output = outputs[:,0,:]
    # # Create the final classification layer
        weights, b, score = fulconn_layer(outputs)
    # score = tf.reshape(score, [input_dim])

    # score1 = tf.to_float(score)

    # score2 = tf.nn.l2_normalize(score1, 0, epsilon=1e-12, name=None)
    score = tf.div(
       tf.subtract(
          score,
          tf.reduce_min(score)
       ),
       tf.subtract(
          tf.reduce_max(score),
          tf.reduce_min(score)
       )
    )
    return score