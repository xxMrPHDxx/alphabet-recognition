from multiprocessing import Pool
from os.path import isfile
from os import listdir, rename
from time import time

def temporary():
	for path in listdir('Datasets'):
		path = f'Datasets/{path}'
		i = 1
		for subpath in listdir(path):
			subpath = f'{path}/{subpath}'
			if isfile(subpath) and subpath[-4:] == '.png':
				name = str(i).rjust(6, '0')
				tmp  = f'{path}/{int(time())}_{i}.tmp.png'
				rename(subpath, tmp)
				yield f'{path}/{name}.png', tmp
				i += 1
	
def cleanup(args):
	name, tmp_name = args
	rename(tmp_name, name)

if __name__ == '__main__':
	pool = Pool(16)
	pool.map(cleanup, temporary())
