from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split

class DataSet(object):
	def __init__(self, videos, dirnames):
		self._num_of_video = len(videos)
		self._dirnames = dirnames
		self._vidoes = videos
		self._index_in_epoch = 0
		self._epochs_completed = 0

	def directoryName(self):
		return self._dirnames

	def videos(self):
		'''Returns images.'''
		return self._vidoes

	def num_examples(self):
		'''Returns number of images.'''
		return self._num_of_video

	def epochs_completed(self):
		'''Returns number of completed epochs.'''
		return self._epochs_completed

	def next_batch(self, batch_size):
		'''Return the next `batch_size` images from the data set.'''
		start = self._index_in_epoch
		self._index_in_epoch += batch_size

		if self._index_in_epoch > self._num_of_video:
			self._epochs_completed += 1
			self._index_in_epoch = 0
			return None, None

			# perm = np.arange(self._num_of_video)
			# np.random.shuffle(perm)
			# self._vidoes = self._vidoes[perm]

			# start = 0
			# self._index_in_epoch = batch_size
			# assert batch_size <= self._num_of_video

		end = self._index_in_epoch

		return self._vidoes[start:end], self._dirnames[start]

def read_images(filenames):
	'''Reads images from file names'''
	images = []
	for file in filenames:
		# you should not resize it manually
		img = Image.open(file).resize((224,224))
		image = np.array(img, dtype = np.float32)
		# image = np.multiply(image, 1.0 / 255.0)
		# image = 2*(image/255.0)-1.0
		images.append(image)
	return images

def read_dataset(path):
	'''Creates data set'''
	dirpath, dirnames, filenames = next(os.walk(path))
	directoryLenght = len(dirnames)
	videoimages = []
	for x in range(0, directoryLenght):
		videodirpath, videodirnames, videofilenames = next(os.walk(path+"/" + dirnames[x]))
		images = read_images([os.path.join(videodirpath, filename) for filename in videofilenames])
		images = np.array(images, dtype = np.float32)
		videoimages.append(images)
	train_images  = videoimages
	return DataSet(train_images, dirnames)