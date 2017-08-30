import tensorflow as tf
import time

import sys
import os, glob
from time import gmtime, strftime


sys.path.insert(0,"./SumMe/python")
#
from demo import *
slim = tf.contrib.slim
from inception_v1 import *
import inception
from utils import read_dataset
import numpy as np
from frame_selector import frame_selector_model
from v_lstm_ae import ConvVAE
from vanila import *
import csv


def guarantee_initialized_variables(session, list_of_variables = None):
    if list_of_variables is None:
        list_of_variables = tf.all_variables()
    uninitialized_variables = list(tf.get_variable(name) for name in
                                   session.run(tf.report_uninitialized_variables(list_of_variables)))
    session.run(tf.initialize_variables(uninitialized_variables))
    return uninitialized_variables

# Number of epochs
num_epochs = 2

# number of video at once
batch_size = 1

#number of frame per video
max_number_of_frames = 32

# Google Save Model
checkpoint_file = 'inception_v1.ckpt'
# path = tf.train.latest_checkpoint('checkpoints')

# input image placeHolder
input_tensor = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name = "input_video_frame_placeholder")

sequence_length = tf.placeholder(tf.int32, shape=[None], name = "input_original_video_sequence_placeholder")

# frame decode to
latent_dim = 100

# percentage of frames to be selected per video
fraction_selection = 0.3

# Load the model
sess = tf.Session()

# default image dimension
inception_v1.default_image_size = 224

# Feature Extraction Scope definition
inception_v1_arg_scope = inception.inception_v1_arg_scope()

# Feature Extraction
with slim.arg_scope(inception_v1_arg_scope):
    extracedFeature = inception_v1(input_tensor, is_training=False, num_classes=1000)
    extracedFeature1 = tf.reshape(extracedFeature, [batch_size, -1, 1024])

saver = tf.train.Saver()

with tf.variable_scope('video_summary'):

    scores = frame_selector_model(extracedFeature1, sequence_length)

    # position of frames with top score
    number_of_frames_selected = tf.cast(fraction_selection * tf.cast(sequence_length[0], tf.float32), tf.int32)

    # values in increasing order but indices is random
    values, indices = tf.nn.top_k(scores, number_of_frames_selected)

    # sort indices and select all
    sortedIndicesValue, sortedindi = tf.nn.top_k(indices, number_of_frames_selected)

    # reverse along 1 (a video)
    reverseindiesDimeOne = tf.reverse(sortedIndicesValue, [1])

    reverseindies = tf.expand_dims(reverseindiesDimeOne, axis=2)

    b = tf.stack([i*tf.ones([number_of_frames_selected], dtype = tf.int32) for i in range(0, reverseindies.shape[0])])
    # b = tf.constant(tf.concat([i*tf.ones([sequence_length[0]])  for i in range(0, reverseindies.shape[0])], axis=0), dtype=tf.int32)
    b = tf.expand_dims(b, 2)

    final_ids = tf.concat([b, reverseindies], axis=2)

    #combine frames with top score
    selectedFramesFeature = tf.gather_nd(extracedFeature1,final_ids)

    #convert to pass to the lstm variational auto encoder
    #load lstm variational auto encoder

    sequence_length_vae = tf.reshape(number_of_frames_selected, [batch_size])
    cvae = ConvVAE(latent_dim, selectedFramesFeature, batch_size, sequence_length_vae)

    ganLoss, D_real, D_fake = gan_loss(selectedFramesFeature, extracedFeature1,  sequence_length, sequence_length_vae)

    lossScore = (tf.reduce_sum(scores)/batch_size*tf.cast(sequence_length[0], tf.float32)) - fraction_selection

    #loss calculation 1 sparsity + (reconst + priop )
    encoderDecoderLoss = lossScore + cvae.loss

    #loss calculation 2 reconst + GAN
    reconsGanLoss = cvae.generation_loss_mean + ganLoss

    #loss calculation 3 GAN (negative sign is added to minimize as only supported by tensorflow
    ganLossAlone = -ganLoss
    # saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='video_summary'))
    saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='video_summary'), max_to_keep = 2)

    #optimizer 1 training variable list
    encoderDecoderTrainList=[tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='video_summary/bi_directional_lstm'), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='video_summary/encoder')]
    #optimizer 1
    print ("optimizer 1 started")
    encoderDecoderTrain = tf.train.AdamOptimizer().minimize(encoderDecoderLoss, var_list=encoderDecoderTrainList)
    print ("optimizer 1 done")

    #optimizer 2 training variable list
    reconsGanLossTrainList=[tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='video_summary/decoder')]
    #optimizer 2
    reconsGanLossTrain = tf.train.AdamOptimizer().minimize(reconsGanLoss, var_list=reconsGanLossTrainList)
    print ("optimizer 2 done")

    #optimizer 3 training variable list
    ganLossAloneTrainList=[tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='video_summary/discriminator')]
    #optimizer 3
    ganLossAloneTrain = tf.train.AdamOptimizer().minimize(ganLossAlone, var_list=ganLossAloneTrainList)
    print ("optimizer 3 done")

    saver.restore(sess, checkpoint_file)
    print ("saver restore done")

    # saver1.restore(sess, path)
    # print("saver restore done")

    images = read_dataset("demodata")
    print ("read dataset done")

    # read dataSet
    tf.report_uninitialized_variables()

    # guarantee_initialized_variables(sess)
    sess.run(tf.global_variables_initializer())

    t = time.time()

    curren_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    checkpoint_measure_dir = "Checkpoints_" + curren_time
    if not os.path.exists(checkpoint_measure_dir):
        os.makedirs(checkpoint_measure_dir)

    #fmeasure saver
    f_measure_dir = "FM_" + curren_time
    if not os.path.exists(f_measure_dir):
        os.makedirs(f_measure_dir)

    while (images.epochs_completed() < num_epochs):
        current_epoch = images.epochs_completed()
        f_measure_epch =[]
        print('[----Epoch {} is started ----]'.format(current_epoch))
        # take next batch until epoch is completed
        i = 0
        csvFileName = f_measure_dir +"/fmeasure_" + str(current_epoch) +".csv"
        f = open(csvFileName, 'w')
        spamwriter = csv.writer(f, delimiter=',')
        while (images.epochs_completed() < current_epoch + 1):
            # get the input images
            input_images, videoName = images.next_batch(batch_size)
            if input_images != None:
                print('    [----Batch {} is started ----]'.format(i))
                numberFrames = np.asarray(input_images).shape[1]
                input_images = np.asarray(input_images).reshape([batch_size*numberFrames, 224, 224, 3])
                # if numberFrames < max_number_of_frames:
                #     input_images = np.append(input_images, np.zeros([max_number_of_frames -numberFrames, 224, 224, 3]), axis=0)
                sequence_length_original_video = np.asarray(numberFrames).reshape(batch_size)
                reverseindiesDimeOne1, encoderDecoderLoss1, reconsGanLoss1,ganLossAlone1,D_real1, D_fake1 ,  _, _, _ = sess.run([ reverseindiesDimeOne, encoderDecoderLoss, reconsGanLoss,ganLossAlone,D_real, D_fake , encoderDecoderTrain, reconsGanLossTrain, ganLossAloneTrain], feed_dict={input_tensor: input_images, sequence_length : sequence_length_original_video})
                f_measure, summary_length = evaluation(videoName, reverseindiesDimeOne1)
                print("        encoderDecoderLoss =", encoderDecoderLoss1 , ", reconsGanLoss =", reconsGanLoss1, ", ganLossAlone =", ganLossAlone1)
                print('    [----Batch {} is finished ----]'.format(i))
                i = i+1
                spamwriter.writerow([current_epoch , videoName , f_measure])
        print('[----Epoch {} is finished ----]'.format(current_epoch))
        saver.save(sess, checkpoint_measure_dir + '/', global_step=current_epoch)
        print ('[----Checkpoint is saved----]')
        f.close()
    print ('Training time: {}s'.format(time.time() - t))

