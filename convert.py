#!/usr/bin/env python
import tensorflow as tf
from model import Model

if __name__ == '__main__':
	model = Model()
	model.load_weights('model').expect_partial()

	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	lite_model = converter.convert()
	with open('model.tflite', 'wb') as f:
		f.write(lite_model)
