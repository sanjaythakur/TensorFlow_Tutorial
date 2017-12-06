import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import sys

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28

# Each image is 28 X 28 in dimension. We flatten the image to get 784 features where each feature would correspond to one pixel.
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Batch size 
BATCH_SIZE = 100

# Defining the number of units in hidden layers
HIDDEN_LAYER_1 = 50
HIDDEN_LAYER_2 = 20

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Allocating nodes in the computational graph to accept inputs. These are means to insert inputs into our computational graph.
with tf.name_scope('inputs'):
	images = tf.placeholder(tf.float32, shape=(None, IMAGE_PIXELS), name='images')
	true_labels = tf.placeholder(tf.int32, shape=(None, NUM_CLASSES), name='true_labels')


# Defining the first hidden layer
with tf.name_scope('first_hidden_layer'):
    weights_layer_1 = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, HIDDEN_LAYER_1],
                          stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name='weights')

    biases_layer_1 = tf.Variable(tf.zeros([HIDDEN_LAYER_1]))

    hidden_output_1 = tf.nn.relu(tf.matmul(images, weights_layer_1) + biases_layer_1)

# Defining the second hidden layer
with tf.name_scope('second_hidden_layer'):
    weights_layer_2 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_1, HIDDEN_LAYER_2],
                          stddev=1.0 / math.sqrt(float(HIDDEN_LAYER_1))))

    biases_layer_2 = tf.Variable(tf.zeros([HIDDEN_LAYER_2]))

    hidden_output_2 = tf.nn.relu(tf.matmul(hidden_output_1, weights_layer_2) + biases_layer_2)

# Defining the outputs
with tf.name_scope('output'):
    weights_output = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_2, NUM_CLASSES],
                          stddev=1.0 / math.sqrt(float(HIDDEN_LAYER_2))))

    biases_output = tf.Variable(tf.zeros([NUM_CLASSES]))

    prediction = tf.matmul(hidden_output_2, weights_output) + biases_output


# Evaluating the loss function
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_labels, logits=prediction), name='loss_operation')


# Creating nodes for summarization that are supposed to be seen through Tensorboard.
with tf.name_scope('summary'):
    loss_tracker = tf.summary.scalar('Loss', loss)
    summary_op = tf.summary.merge_all()



# Updating the parameters
optimizer = tf.train.GradientDescentOptimizer(0.05)
training = optimizer.minimize(loss)

# Create a Saver object
saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)

previous_minimum_loss = sys.float_info.max

with tf.Session() as sess:
    
    init = tf.global_variables_initializer()
    sess.run(init)

    # This line tells our program where to save the files that keep track of any information viewable via tensorboard
    location_to_store_at = './graphs/'

    writer = tf.summary.FileWriter(location_to_store_at, sess.graph)


    for epoch_iterator in range(100):
        batch = mnist.train.next_batch(BATCH_SIZE)
        loss_summary, loss_value, _ = sess.run([summary_op, loss, training], {images: batch[0], true_labels: batch[1]})            
        if loss_value < previous_minimum_loss:
            saver.save(sess, './model_iter/model_iter', global_step = epoch_iterator, write_meta_graph=False)
            previous_minimum_loss = loss_value
        writer.add_summary(loss_summary, global_step=epoch_iterator)

    writer.close()
    # Save the final model
    saver.save(sess, './model_final/model_final', write_state=False)
    
    # Finally we test how good our final model is by checking the loss incurred on the previously unseen data.
    print('The loss incurred on test set is ' + str(sess.run([loss], {images:mnist.test.images, true_labels: mnist.test.labels})))