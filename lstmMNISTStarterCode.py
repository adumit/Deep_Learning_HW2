import tensorflow as tf 
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np 

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)#call mnist function

learningRate = 0.001
trainingIters = 100000
batchSize = 128
displayStep = 10

nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
nHidden = 256 #number of neurons for the RNN
nClasses = 10 #this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])
istate = tf.placeholder("float", [None, 2*nHidden])

weights = {
    'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
    'out': tf.Variable(tf.random_normal([nClasses]))
}

typeCell = 'lstm'

def RNN(x, W, B):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(0, nSteps, x) #configuring so you can get it as needed for the 28 pixels

    if typeCell == 'basic':
        lstmCell = rnn_cell.BasicRNNCell(nHidden)
    elif typeCell == 'lstm':
        lstmCell = rnn_cell.LSTMCell(nHidden)
    elif typeCell == 'gru':
        lstmCell = rnn_cell.GRUCell(nHidden)
    else:
        raise Exception("Bad typeCell value!")

    outputs, states = rnn.rnn(lstmCell, x, dtype=tf.float32)#for the rnn where to get the output and hidden state

    return tf.matmul(outputs[-1], W['out']) + B['out']

pred = RNN(x, weights, biases)


# optimization
# create the cost, optimization, evaluation, and accuracy
# for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init)
    step = 1

    testData = mnist.test.images.reshape((-1, nSteps, nInput))
    testLabel = mnist.test.labels

    train_errors = []
    train_loss = []
    test_errors = []

    while step * batchSize < trainingIters:
        batchX, batchY = mnist.train.next_batch(batchSize)#mnist has a way to get the next batch
        batchX = batchX.reshape((batchSize, nSteps, nInput))

        sess.run(optimizer, feed_dict={x: batchX, y: batchY,
                                       istate: np.zeros((batchSize, 2*nHidden))})

        if step % displayStep == 0:
            acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY,
                                                istate: np.zeros((batchSize, 2*nHidden))})
            loss = sess.run(cost, feed_dict={x: batchX, y: batchY,
                                             istate: np.zeros((batchSize, 2*nHidden))})
            print("Iter " + str(step*batchSize) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))
            test_err = sess.run(accuracy, feed_dict={x: testData, y: testLabel,
                                                istate: np.zeros(
                                                    (batchSize, 2 * nHidden))})
            print("Testing Accuracy: " + str(test_err))

            test_errors.append(test_err)
            train_errors.append(acc)
            train_loss.append(loss)

        step += 1
    print('Optimization finished')

    with open('recurrentNN_errors.txt', 'a') as f:
        f.write("Run with parameters: \n")
        f.write("Num Hidden: " + str(nHidden) + '\n')
        f.write("Num Iterations: " + str(trainingIters) + '\n')
        f.write("Type of cell: " + str(typeCell) + '\n')
        f.write("Train errors: " + ",".join([str(x) for x in train_errors]) + "\n")
        f.write("Train loss: " + ",".join([str(x) for x in train_loss]) + "\n")
        f.write("Test errors: " + ",".join([str(x) for x in test_errors]) + '\n')
        f.write('\n\n')

