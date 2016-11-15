from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib as mil
import os
mil.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import time

# --------------------------------------------------
# setup

def weight_variable(shape, name):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    W = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return W

def bias_variable(shape, name):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    b = tf.Variable(tf.constant(0.1, shape=shape), name=name)
    return b


def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv


def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return h_max


def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def variable_summaries(var, name):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)


ntrain = 1000 # per class
ntest = 100 # per class
nclass =  10 # number of classes
imsize = 28
nchannels = 1
batchsize = 500

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = 'CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

tf_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='tf_data')#tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_labels = tf.placeholder(tf.float32, shape=[None, 10], name='tf_labels')#tf variable for labels

# --------------------------------------------------
# model
#create your model
is_training = tf.placeholder(tf.bool)

# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')
b_conv1 = bias_variable([32], name='b_conv1')
h_prerelu1 = conv2d(tf_data, W_conv1)
h_bn1 = batch_norm(h_prerelu1, 32, is_training)
h_conv1 = tf.nn.relu(h_bn1 + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')
b_conv2 = bias_variable([64], name='b_conv2')
h_prerelu2 = conv2d(h_pool1, W_conv2)
h_bn2 = batch_norm(h_prerelu2, 64, is_training)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')
b_fc1 = bias_variable([1024], name='b_fc1')
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax
W_fc2 = weight_variable([1024, 10], name='W_fc2')
b_fc2 = bias_variable([10], name='b_fc2')
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# --------------------------------------------------
# Visualizers
# Functions sourced from: https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4#.io2340g7d
def getActivations(layer,stimuli,plot_weights=False):
    if plot_weights:
        units = layer.eval(session=sess)
        plotWeights(units)
    else:
        units = layer.eval(session=sess,feed_dict={tf_data:np.reshape(stimuli,[1, 28,28, 1],order='F'),keep_prob:1.0,is_training:False})
        plotNNFilter(units)

def plotWeights(units):
    filters = 32
    plt.figure(1, figsize=(30, 30))
    print(units.shape)
    for i in range(0, filters):
        plt.subplot(7, 6, i + 1)
        plt.imshow(units[:, :, 0, i], interpolation="nearest", cmap = plt.get_cmap('gray'))
    plt.show()

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20, 20))
    for i in range(0, filters):
        plt.subplot(7, 6, i + 1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0, :, :, i], interpolation="nearest", cmap = plt.get_cmap('gray'))
    plt.show()
# -------------------


variable_summaries(W_conv1, 'conv_layer1/weights')
tf.histogram_summary('conv_layer1/weight_hist', W_conv1)
variable_summaries(b_conv1, 'conv_layer1/biases')
tf.histogram_summary('conv_layer1/bias_hist', b_conv1)
variable_summaries(h_conv1, 'conv_layer1/activation')
tf.histogram_summary('conv_layer1/activation_hist', h_conv1)
variable_summaries(h_pool1, 'conv_layer1/postpool')
tf.histogram_summary('conv_layer1/postpool_hist', h_pool1)

variable_summaries(W_conv2, 'conv_layer2/weights')
tf.histogram_summary('conv_layer2/weight_hist', W_conv2)
variable_summaries(b_conv2, 'conv_layer2/biases')
tf.histogram_summary('conv_layer2/bias_hist', b_conv2)
variable_summaries(h_conv2, 'conv_layer2/activation')
tf.histogram_summary('conv_layer2/activation_hist', h_conv2)
variable_summaries(h_pool2, 'conv_layer2/postpool')
tf.histogram_summary('conv_layer2/postpool_hist', h_pool2)

# --------------------------------------------------
# loss
#set up the loss, optimization, evaluation, and accuracy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_labels * tf.log(y_conv), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --------------------------------------------------
# optimization
saver = tf.train.Saver()

def check_for_load_model():
    load_checkpoint = input(
        "Do you want to load a checkpoint file? (Enter path to checkpoint file or nothing for no checkpoint)")
    sess.run(tf.initialize_all_variables())

    if load_checkpoint == "":
        print("Starting to train from scratch...")
    else:
        file_found = False
        while file_found == False or load_checkpoint == "":
            saver.restore(sess, load_checkpoint)
            try:
                print("Loading model...")

                file_found = True
            except e:
                print("That file path was not found...")
                load_checkpoint = input("Enter the path to try again:")

if __name__ == "__main__":
    with tf.Session() as sess:

        what_to_do = input("Train or visualize the network?")

        if what_to_do == "train" or what_to_do == "t":
            check_for_load_model()

            batch_xs = np.zeros([batchsize, imsize, imsize,
                                 nchannels])  # setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
            batch_ys = np.zeros(
                [batchsize, nclass])  # setup as [batchsize, the how many classes]

            train_errors = []
            test_errors = []
            result_dir = './results/'
            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(result_dir, sess.graph)
            for i in range(2001): # try a small iteration size once it works then continue
                perm = np.arange(ntrain*nclass)
                np.random.shuffle(perm)
                for j in range(batchsize):
                    batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
                    batch_ys[j,:] = LTrain[perm[j],:]
                if i%5 == 0:
                    # calculate train accuracy and print it
                    train_err = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0, is_training: False})
                    test_err = accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0, is_training: False})

                    train_errors.append(train_err)
                    test_errors.append(test_err)

                    print("On iteration: " + str(i) + ", train accuracy was %g" % train_err)
                    print("test accuracy %g" % test_err)

                if i%50 == 0:
                    summary_str = sess.run(summary_op, feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5, is_training: True})
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()
                if i%200 == 0 and i != 0:
                    checkpoint_file = os.path.join(result_dir, 'checkpoint')
                    saver.save(sess, checkpoint_file, global_step=i)

                if train_errors[-10:] == [1,1,1,1,1,1,1,1,1,1]:
                    break
                optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5, is_training: True}) # dropout only during training

            with open('train_and_test_errors.txt', 'w') as f:
                f.write(",".join([str(x) for x in train_errors]) + "\n")
                f.write(",".join([str(x) for x in test_errors]))

            # --------------------------------------------------
            # test
            print("Final Test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0, is_training: False}))

            sess.close()

        elif what_to_do == "visualize" or what_to_do == "v":
            check_for_load_model()
            while True:
                class_to_viz = input("Which test class would you like to visualize? (0 to 9)")
                image_to_viz = input("Favorite number for 0 to 99?")
                image_to_use = "/CIFAR10/Test/" + str(class_to_viz) + "/Image000" + str(image_to_viz) + ".png"
                im = misc.imread(path); # 28 by 28
                im = im.astype(float)/255
                print("Here's the image...")
                plt.imshow(np.reshape(im,[28,28]), interpolation="nearest", cmap = plt.get_cmap('gray'))
                plt.show()
                print("Here are the activation filters from the first layer...")
                getActivations(W_conv1, im, True)

