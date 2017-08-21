import tensorflow as tf
from ops import *
import numpy as np
import matplotlib.pyplot as plt


class ConvVAE(object):
    '''Convolutional variational autoencoder'''
    def __init__(self, latent_dim, input_images, batch_size, number_of_frames):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lstm_hidden_layer = 2
        self.lstm_hidden_units = 2048
        self.number_of_frames = number_of_frames

        # placeholder for input images. Input images are RGB 64x64
        self.input_images = input_images
        p_inputs = [tf.squeeze(t, [1]) for t in tf.split(self.input_images, self.number_of_frames, 1)]
        input_images_flat = tf.reshape(self.input_images, [batch_size, -1, 1024])

        # placeholder for z_samples. We are using this placeholder when we are generating new images
        # self.z_samples = tf.placeholder(tf.float32, [batch_size, number_of_frames, self.latent_dim], name="generate_placeholder")
        # z_samples2 = [tf.squeeze(t, [1]) for t in tf.split(self.z_samples, number_of_frames, 1)]

        # self.z_samples = tf.placeholder(tf.float32, [None, 32, 32], name="generate_placeholder")
        # p_inputs_1 = [tf.squeeze(t, [1]) for t in tf.split(self.z_samples, 32, 1)]

        stacked_rnn = []
        for iiLyr in range(self.lstm_hidden_layer):
            stacked_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=self.lstm_hidden_units, state_is_tuple=True))
        lstm_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)

        stacked_rnn1 = []
        for iiLyr in range(self.lstm_hidden_layer):
            stacked_rnn1.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=self.lstm_hidden_units, state_is_tuple=True))
        lstm_multi_cell1 = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn1)

        self._enc_cell = lstm_multi_cell
        self._dec_cell = lstm_multi_cell1

        # encoder output
        z_mean, z_logstd = self.encoder(p_inputs)

        # decoder input
        samples = tf.random_normal([self.batch_size, self.number_of_frames, self.latent_dim], 0, 1, dtype=tf.float32)
        z = z_mean + (tf.exp(.5 * z_logstd) * samples)
        z = [tf.squeeze(t, [1]) for t in tf.split(z, number_of_frames, 1)]

        # decoder output
        self.generated_images = self.decoder(z, None)
        self.generated_images_sigmoid = tf.sigmoid(self.generated_images)
        generated_images_flat = self.generated_images

        # let's calculate the loss
        '''
        self.generation_loss = -tf.reduce_sum(input_images_flat * tf.log(1e-8 + generated_images_flat)\
                                         + (1 - input_images_flat) * tf.log(1e-8 + 1 - generated_images_flat), 1)'''

        self.generation_loss = tf.reduce_sum(tf.maximum(generated_images_flat, 0) - generated_images_flat * input_images_flat + tf.log(1 + tf.exp(-tf.abs(generated_images_flat))), 2)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.exp(2 * z_logstd) - 2 * z_logstd - 1, 2)

        self.generation_loss_mean = tf.reduce_mean(self.generation_loss)

        self.loss = tf.reduce_mean(self.generation_loss + self.latent_loss)
        # and our optimizer
        learning_rate = 1e-3
        # self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        # generator for new frames
        # self.generator = self.decoder(z_samples2, True, activation=tf.nn.sigmoid)

        # self.generator = tf.random_uniform([100, 50])

    def encoder(self, data):

        with tf.variable_scope('encoder') as vs:
            z_codes, self.enc_state = tf.contrib.rnn.static_rnn(
                self._enc_cell, data, dtype=tf.float32)

            batch_num = data[0].get_shape().as_list()[0]
            elem_num = data[0].get_shape().as_list()[1]

        # first convolutional layer 64x64x3 -> 32x32x16

        # h1 = tf.nn.relu(conv2d(z_codes[-1], 3, 16, 'conv1'))
        #
        # # second convolutional layer 32x32x16 -> 16x16x32
        # h2 = tf.nn.relu(conv2d(h1, 16, 32, 'conv2'))
        #
        # fully connected layer
        # h2_flat = tf.reshape(z_codes[-1], [-1, 16 * 16 * 32])
        #

        # fully connected layer
        #     h2_flat = z_codes[0]
            h2_flat = tf.transpose(tf.stack(z_codes), [1, 0, 2])
            #
            # z_mean = dense(h2_flat, 16 * 16 * 32, self.latent_dim, 'z_mean_dense')
            # z_logstd = dense(h2_flat, 16 * 16 * 32, self.latent_dim, 'z_stddev_dense')

            z_mean = dense(h2_flat, self.batch_size, self.lstm_hidden_units, self.latent_dim, 'z_mean_dense')
            z_logstd = dense(h2_flat, self.batch_size, self.lstm_hidden_units, self.latent_dim, 'z_stddev_dense')

        return z_mean, z_logstd
        # return  z_codes

    def decoder(self, z,reuse, activation=tf.identity,  reverse=True):

        # z = tf.expand_dims(z, 0)
        # z = tf.tile(z, [time_step,1,1])
        # z = tf.unstack(z)

        with tf.variable_scope('decoder', reuse=reuse):
            # batch_num = z[0].get_shape().as_list()[0]
            # elem_num = z[0].get_shape().as_list()[1]

            dec_outputs, dec_state = tf.contrib.rnn.static_rnn(
                self._dec_cell, z, initial_state=self.enc_state, dtype=tf.float32)

            if reverse:
                dec_outputs = dec_outputs[::-1]
            dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])

            dec_weight_ = tf.get_variable("dec_weight",shape=[self.lstm_hidden_units, 1024], initializer=tf.contrib.layers.xavier_initializer())

            dec_bias_ = tf.get_variable("dec_bias",
                                    initializer = tf.zeros(shape=[1024], dtype=tf.float32))

            expaned = tf.expand_dims(dec_weight_, 0)

            dec_weight_ = tf.tile(expaned,[self.batch_size, 1, 1])
            output = tf.matmul(dec_output_, dec_weight_) + dec_bias_


            # dec_state = self.enc_state
            # dec_input_ = tf.zeros(tf.shape(hey), dtype=tf.float32)
            # dec_outputs = []
            # for step in range(len(z)):
            #     if step > 0: vs.reuse_variables()
            #     dec_input_, dec_state = self._dec_cell(dec_input_, dec_state)
            #     dec_input_ = tf.matmul(dec_input_, dec_weight_) + dec_bias_
            #     dec_outputs.append(dec_input_)
            # if reverse:
            #     dec_outputs = dec_outputs[::-1]
            # output = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])

            # h1 = tf.nn.relu(deconv2d(z_matrix, [self.batch_size, 32, 32, 16], 'deconv1', reuse))

            # second deconvolutional layer 32x32x16 -> 64x64x3
            # h2 = deconv2d(h1, [self.batch_size, 64, 64, 3], 'deconv2', reuse)

        return activation(output)

    # def training_step(self, sess, input_images):
    #     sess.run(self.optimizer, feed_dict={self.input_images: input_images})

    # def loss_step(self, sess, input_images):
    #     return sess.run(self.loss, feed_dict={self.input_images: input_images})

    def generation_step(self, z_samples):
        '''Generates new images'''
        return self.decoder(z_samples, True, activation=tf.nn.sigmoid)

    # def recognition_step(self, sess, input_images):
    #     '''Reconstruct images'''
    #     return sess.run(self.generated_images_sigmoid, feed_dict={self.input_images: input_images})


# if __name__ == '__main__':
#     # Let's test it before use
#     cvae = ConvVAE(100, batch_size=5)
#     init = tf.global_variables_initializer()
#
#     z_sample = np.random.normal(size=(5,32,1024))
#
#     with tf.Session() as sess:
#         sess.run(init)
        # cvae.training_step(sess, z_sample)
        # loss = cvae.loss_step(sess, z_sample)
        # print(loss)
        # output_frame = output_frame * 255
        # output_frame = output_frame.astype(np.uint8)
        # print('Shape= ', output_frame.shape)
        # plt.imshow(np.reshape(output_frame, [64, 64, 3]))
        # plt.show()

    # cvae = ConvVAE(100, batch_size=1)
    # init = tf.global_variables_initializer()
    #
    # z_sample = np.random.normal(size=100)
    #
    # # print('z= ', z_sample)
    #
    # with tf.Session() as sess:
    #     sess.run(init)
    #     output_frame = cvae.generation_step(sess, np.reshape(z_sample, [1, 100]))
    #     print (output_frame)
    #     # output_frame = output_frame * 255
    #     # output_frame = output_frame.astype(np.uint8)
    #     # print('Shape= ', output_frame.shape)
    #     # plt.imshow(np.reshape(output_frame, [64, 64, 3]))
    #     # plt.show()
