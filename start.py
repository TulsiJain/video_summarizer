import tensorflow as tf
import time

slim = tf.contrib.slim
from inception_v1 import *
import inception
from utils import read_dataset
import numpy as np
from frame_selector import frame_selector_model
from v_lstm_ae import ConvVAE
from vanila import *

# Number of epochs
num_epochs = 5

# number of video at once
batch_size = 1

#number of frame per video
number_of_frames = 9892

# Google Save Model
checkpoint_file = 'inception_v1.ckpt'

# input image placeHolder
input_tensor = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])

# frame decode to
latent_dim = 100

# percentage of frames to be selected per video
fraction_selection = 0.2

# Load the model
sess = tf.Session()

# default image dimension
inception_v1.default_image_size = 224

# Feature Extraction Scope definition
inception_v1_arg_scope = inception.inception_v1_arg_scope()

# Feature Extraction
with slim.arg_scope(inception_v1_arg_scope):
    extracedFeature = inception_v1(input_tensor, is_training=False, num_classes=1000)
    extracedFeature1 = tf.reshape(extracedFeature, [batch_size, number_of_frames, 1024])

saver = tf.train.Saver()

# frame_input = tf.placeholder(tf.float32, shape=[None, 32, 32])
outputs, weights, biased, scores = frame_selector_model(extracedFeature1)

# position of frames with top score
number_of_frames_selected = int(fraction_selection * number_of_frames)

# values in increasing order but indices is random
values, indices = tf.nn.top_k(scores, number_of_frames_selected)

# sort indices and select all
sortedIndicesValue, sortedindi = tf.nn.top_k(indices, number_of_frames_selected)

# reverse along 1 (a video)
reverseindies = tf.reverse(sortedIndicesValue, [1])

reverseindies = tf.expand_dims(reverseindies, axis=2)

b = tf.constant(np.asarray([i*np.ones(reverseindies.shape[1]) for i in range(0, reverseindies.shape[0])], dtype=np.int32), dtype=tf.int32)
b = tf.expand_dims(b, 2)

final_ids = tf.concat([b, reverseindies], axis=2)

#combine frames with top score
selectedFramesFeature = tf.gather_nd(extracedFeature1,final_ids)

#convert to pass to the lstm variational auto encoder
#load lstm variational auto encoder
cvae = ConvVAE(latent_dim, selectedFramesFeature, batch_size, number_of_frames_selected)

ganLoss = gan_loss(selectedFramesFeature, extracedFeature1)

lossScore = (tf.reduce_sum(scores)/batch_size*number_of_frames) - fraction_selection

#loss calculation 1 sparsity + (reconst + priop )
encoderDecoderLoss = lossScore + cvae.loss

#loss calculation 2 reconst + GAN
reconsGanLoss = cvae.generation_loss_mean + ganLoss

#loss calculation 3 GAN (negative sign is added to minimize as only supported by tensorflow
ganLossAlone = -ganLoss

#optimizer 1 training variable list
encoderDecoderTrainList=[tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='bi_directional_lstm'), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')]

#optimizer 1
encoderDecoderTrain = tf.train.AdamOptimizer().minimize(encoderDecoderLoss, var_list=encoderDecoderTrainList)

#optimizer 2 training variable list
reconsGanLossTrainList=[tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')]

#optimizer 2
reconsGanLossTrain = tf.train.AdamOptimizer().minimize(reconsGanLoss, var_list=reconsGanLossTrainList)

#optimizer 3 training variable list
ganLossAloneTrainList=[tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')]

#optimizer 3
ganLossAloneTrain = tf.train.AdamOptimizer().minimize(ganLossAlone, var_list=ganLossAloneTrainList)

saver.restore(sess, checkpoint_file)

# read dataSet
images = read_dataset("data")

sess.run(tf.global_variables_initializer())

t = time.time()

while (images.epochs_completed() < num_epochs):

    current_epoch = images.epochs_completed()
    print('[----Epoch {} is started ----]'.format(current_epoch))
    # take next batch until epoch is completed
    i =0
    while (images.epochs_completed() < current_epoch + 1):
        # get the input images
        input_images = images.next_batch(batch_size)
        if input_images != None:
            print('    [----Batch {} is started ----]'.format(i))
            input_images = np.asarray(input_images).reshape([batch_size*number_of_frames, 224, 224, 3])
            extracedFeature2, outputs1, weights1, biased1, scores1, encoderDecoderLoss1, reconsGanLoss1,ganLossAlone1, _, _, _ = sess.run([extracedFeature1, outputs, weights, biased, scores, encoderDecoderLoss, reconsGanLoss, ganLossAlone, encoderDecoderTrain, reconsGanLossTrain, ganLossAloneTrain  ], feed_dict={input_tensor: input_images})
            print("        encoderDecoderLoss =", encoderDecoderLoss1 , ", reconsGanLoss =", reconsGanLoss1, ", ganLossAlone =", ganLossAlone1)
            print('    [----Batch {} is finished ----]'.format(i))
        i = i + 1
    print('[----Epoch {} is finished ----]'.format(current_epoch))

    # saver.save(sess, 'checkpoints/', global_step=current_epoch)
    # print '[----Checkpoint is saved----]'

print ('Training time: {}s'.format(time.time() - t))
