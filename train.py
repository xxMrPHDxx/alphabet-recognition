import tensorflow as tf
import numpy as np
import pickle

from model import Model

if __name__ == '__main__':
	LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

	with open('x_train.pickle', 'rb') as f:
		X_train = pickle.loads(f.read())
	with open('y_train.pickle', 'rb') as f:
		Y_train = pickle.loads(f.read())
	with open('x_test.pickle', 'rb') as f:
		X_test = pickle.loads(f.read())
	with open('y_test.pickle', 'rb') as f:
		Y_test = pickle.loads(f.read())

	print('Training: ', X_train.shape, Y_train.shape)
	print('Validation: ', X_test.shape, Y_test.shape)

	model = Model()
	model.summary()
	model.fit(
		x=X_train, 
		y=Y_train,
		epochs=10,
		# validation_split=0.05,
		validation_data=(X_test, Y_test),
		workers=32,
		use_multiprocessing=True
	)

	model.save_weights('model')
