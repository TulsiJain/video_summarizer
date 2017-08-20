import tensorflow as tf
import numpy as np

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)



rnn_size = 1024
Z_dim = 100
rnn_layer = 2
batch_size = 2

# X = tf.placeholder(tf.float32, shape=[None, num_of_frames, 1024])
#
# Z = tf.placeholder(tf.float32, shape=[None,num_of_frames, 100])
# z = [tf.squeeze(t, [1]) for t in tf.split(Z, 32, 1)]



# def generator(z, activation=tf.identity, reuse=False, reverse = True):
#         stacked_rnn1 = []
#         for iiLyr in range(2):
#             stacked_rnn1.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=2048, state_is_tuple=True))
#         lstm_multi_cell1 = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn1)
#
#         with tf.variable_scope('decoder') as vs:
#             dec_outputs, dec_state = tf.contrib.rnn.static_rnn(
#                 lstm_multi_cell1, z, dtype=tf.float32)
#
#             if reverse:
#                 dec_outputs = dec_outputs[::-1]
#             dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])
#
#             dec_weight_ = tf.get_variable("dec_weight",
#                         initializer = tf.truncated_normal([2048, 1024], dtype=tf.float32))
#
#             dec_bias_ = tf.get_variable("dec_bias",
#                                     initializer = tf.constant(0.1, shape=[1024], dtype=tf.float32))
#             expaned = tf.expand_dims(dec_weight_, 0)
#
#             dec_weight_ = tf.tile(expaned,[batch_size, 1, 1])
#             output = tf.matmul(dec_output_, dec_weight_) + dec_bias_
#         return activation(output)

def discriminator(x, reuse):

    x = [tf.squeeze(t, [1]) for t in tf.split(x, x.get_shape()[1], 1)]
    # with tf.variable_scope('cell_def'):
    with tf.variable_scope('discriminator', reuse=reuse):
        stacked_rnn1 = []
        for iiLyr1 in range(rnn_layer):
            stacked_rnn1.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size, state_is_tuple=True, reuse=reuse))
        lstm_multi_fw_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn1)

        # with tf.variable_scope('rnn_def'):
        dec_outputs, dec_state = tf.contrib.rnn.static_rnn(
            lstm_multi_fw_cell, x, dtype=tf.float32)

        D_W1 = tf.Variable(xavier_init([1024, 128]))
        D_b1 = tf.Variable(tf.zeros(shape=[128]))

        D_W2 = tf.Variable(xavier_init([128, 1]))
        D_b2 = tf.Variable(tf.zeros(shape=[1]))

        D_h1 = tf.nn.relu(tf.matmul(dec_outputs[-1], D_W1) + D_b1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2

        D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

def gan_loss(G_sample, X):
    # G_sample = generator(z)
    D_real, D_logit_real = discriminator(X,  None)
    D_fake, D_logit_fake = discriminator(G_sample, True)

    D_loss = (tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))  + tf.reduce_mean(tf.log(D_fake)))

    D_loss = tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake) + tf.log(D_fake))


    return D_loss

    # D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)


# merged_summary_op = tf.summary.merge_all()




