import tensorflow as tf

class Model(tf.keras.models.Sequential):
	def __init__(self):
		tf.keras.models.Sequential.__init__(self)
		self.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
		self.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
		self.add(tf.keras.layers.MaxPooling2D((2, 2)))
		self.add(tf.keras.layers.Flatten())
		self.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
		self.add(tf.keras.layers.Dense(26, activation='softmax'))
		self.compile(
			optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9), 
			loss='categorical_crossentropy', 
			metrics=['accuracy']
		)

if __name__ == '__main__':
	model = Model()
	model.load_weights('model')
	model.save('model.h5')
	# print(model.summary())
	# weights = model.get_weights()
	# print([i.shape for i in weights])