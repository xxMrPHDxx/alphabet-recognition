from PIL import Image
import os

def list_folders(path):
	for subpath in os.listdir(path):
		subpath = f'{path}/{subpath}'
		if os.path.isdir(subpath): yield subpath
		
def list_files(path):
	for subpath in os.listdir(path):
		subpath = f'{path}/{subpath}'
		if os.path.isfile(subpath): yield subpath
		
def get_images():
	for folder in list_folders('Datasets'):
		for fname in list_files(folder):
			yield fname
	
def check_size(fname):
	size = Image.open(fname).size
	assert size[0] == size[1] and size[0] == 28, f'Size of {fname} is not 28x28'

if __name__ == '__main__':
	for image_path in get_images():
		check_size(image_path)
