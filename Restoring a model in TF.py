import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


tf.reset_default_graph()  
imported_meta = tf.train.import_meta_graph("./model_final/model_final.meta")

with tf.Session() as sess:  
	imported_meta.restore(sess, tf.train.latest_checkpoint('./model_iter/'))
	graph = tf.get_default_graph()

	images = graph.get_tensor_by_name('inputs/images:0')
	true_labels = graph.get_tensor_by_name('inputs/true_labels:0')
	loss = graph.get_tensor_by_name('loss/loss_operation:0')
	
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	loss_value = sess.run(loss, {images:mnist.test.images, true_labels: mnist.test.labels})
	
	print("Loss: %.8f" % loss_value)