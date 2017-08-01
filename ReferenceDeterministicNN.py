def __init__(self):

		tf.reset_default_graph()

		with tf.name_scope('inputs'):
			self.inputs = tf.placeholder(tf.float32, shape=(None, NUMBER_FEATURES), name='X')
			self.true_labels = tf.placeholder(tf.float32, shape=(None, NUMBER_OUTPUT_UNITS), name='labels')

		
		with tf.name_scope('architecture'):
			
			with tf.name_scope('hidden_layer_1'):
				#standard_deviations = np.full((NUMBER_FEATURES, NUMBER_HIDDEN_UNITS_LAYER_1), 1/PRECISION_ALPHA)

				weights = tf.Variable(tf.truncated_normal([NUMBER_FEATURES, NUMBER_HIDDEN_UNITS_LAYER_1], stddev = 1.0/math.sqrt(float(NUMBER_FEATURES))), name = 'weights')
				biases = tf.Variable(tf.zeros([NUMBER_HIDDEN_UNITS_LAYER_1]), name = 'biases')
				output_from_layer_1 = tf.nn.relu(tf.matmul(self.inputs, weights) + biases) 

			with tf.name_scope('hidden_layer_2'):
				#standard_deviations = np.full((NUMBER_HIDDEN_UNITS_LAYER_1, NUMBER_HIDDEN_UNITS_LAYER_2), 1/PRECISION_ALPHA)

				weights = tf.Variable(tf.truncated_normal([NUMBER_HIDDEN_UNITS_LAYER_1, NUMBER_HIDDEN_UNITS_LAYER_2], stddev = 1.0/math.sqrt(float(NUMBER_HIDDEN_UNITS_LAYER_1))), name = 'weights')
				biases = tf.Variable(tf.zeros([NUMBER_HIDDEN_UNITS_LAYER_2]), name = 'biases')
				output_from_layer_2 = tf.nn.relu(tf.matmul(output_from_layer_1, weights) + biases) 

			with tf.name_scope('output_layer'):
				#standard_deviations = np.full((NUMBER_HIDDEN_UNITS_LAYER_2, NUMBER_OUTPUT_UNITS), 1/PRECISION_ALPHA)
				
				weights = tf.Variable(tf.truncated_normal([NUMBER_HIDDEN_UNITS_LAYER_2, NUMBER_OUTPUT_UNITS], stddev = 1.0/math.sqrt(float(NUMBER_HIDDEN_UNITS_LAYER_2))), name = 'weights')
				biases = tf.Variable(tf.zeros([NUMBER_OUTPUT_UNITS]), name = 'biases')
				self.output = tf.nn.relu(tf.matmul(output_from_layer_2, weights) + biases, name = 'output')


	def predict(self, X):

		with tf.Session() as sess:
			writer = tf.summary.FileWriter('./graphs', sess.graph)
			sess.run(tf.global_variables_initializer())

			# input_tensor = tf.get_default_graph().get_tensor_by_name('inputs/X:0')
			# output_tensor = tf.get_default_graph().get_tensor_by_name('model/output:0')

			prediction = sess.run(self.output, {self.inputs: X})

		writer.close()
		return prediction