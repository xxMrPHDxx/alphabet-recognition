from tensorflow.keras.utils import to_categorical
from tensorflow import convert_to_tensor
from multiprocessing import Pool
from os.path import isfile
from os import listdir
from PIL import Image
import numpy as np
import pickle

ALPHABETS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def get_files():
	for subfolder in listdir('Datasets'):
		alphabet = subfolder
		subfolder = f'Datasets/{subfolder}'
		if isfile(subfolder): continue
		for file in listdir(subfolder):
			file = f'{subfolder}/{file}'
			if isfile(file) and file[-4:] == '.png':
				yield alphabet, file
	
def get_samples(train_ratio=0.9):
	letters          = {}
	for alphabet, file in get_files():
		if alphabet in letters:
			letters[alphabet].append(file)
		else:
			letters[alphabet] = [file]
	train, test = [], []
	for letter in ALPHABETS:
		arr  = letters[letter]
		size = int(len(arr)*train_ratio)
		train += arr[:size]
		test  += arr[size:]
	return train, test

def get_input(path):
	pix = Image.open(path).load()
	return [
    [
      [ sum(pix[x, y][:3])/(255*3) ]
      for x in range(28)
    ]
    for y in range(28)
  ]
		
def get_label(path):
	return ALPHABETS.index(path.split('/')[-2])

if __name__ == '__main__':
	pool = Pool(16)

	train_data,	test_data = get_samples(0.99)

	X_train = np.array(pool.map(get_input, train_data))
	print('Pickling "x_train.pickle"')
	with open('x_train.pickle', 'wb') as f:
		f.write(pickle.dumps(X_train))
		del X_train
	
	Y_train = np.array(pool.map(get_label, train_data))
	Y_train = to_categorical(Y_train)
	print('Pickling "y_train.pickle"')
	with open('y_train.pickle', 'wb') as f:
		f.write(pickle.dumps(Y_train))
		del Y_train

	X_test = np.array(pool.map(get_input, test_data))
	print('Pickling "x_test.pickle"')
	with open('x_test.pickle', 'wb') as f:
		f.write(pickle.dumps(X_test))
		del X_test
	
	Y_test = np.array(pool.map(get_label, test_data))
	Y_test = to_categorical(Y_test)
	print('Pickling "y_test.pickle"')
	with open('y_test.pickle', 'wb') as f:
		f.write(pickle.dumps(Y_test))
		del Y_test
