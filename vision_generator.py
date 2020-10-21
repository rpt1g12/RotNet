import abc

import cv2
from typing import Tuple

import numpy as np
from Vision import Sample
from Vision.io_managers import Manager, SampleAugmenter
from Vision.sample import Sample_List, Sample_Generator
from keras.utils import to_categorical
from keras_preprocessing.image import Iterator


def gray_preprocessor(sample: Sample, theta: float, input_shape, crop) -> Sample:
    sample.set_img(cv2.cvtColor(sample.get_img_arr(), cv2.COLOR_BGR2GRAY))
    return sample.set_rotation(theta).apply_rotation(crop).resize(input_shape)


def color_preprocessor(sample: Sample, theta: float, input_shape, crop) -> Sample:
    return sample.set_rotation(theta).apply_rotation(crop).resize(input_shape)


class VocRotGenerator(Manager):
    """Docstring"""

    def __init__(self, manager: Manager,
                 input_shape: Tuple[int, int],
                 deg_resolution: int = 2,
                 batch_size=32,
                 rotate=-1,
                 preprocessing_function=None,
                 crop=True,
                 n_aug=0,
                 make_grayscale: bool = False,
                 shuffle=True,
                 seed=0):
        self.input_shape = input_shape
        assert 360 % deg_resolution == 0, "Resolution must be a factor of 360"
        self.deg_resolution = deg_resolution
        self.batch_size = batch_size
        self.color_channels = 1 if make_grayscale else 3
        self.crop = crop
        if rotate > 0:
            assert rotate % deg_resolution == 0, "Resolution must be a factor of rotate"
        self.rotate = rotate
        self.make_grayscale = make_grayscale
        if type(manager) != SampleAugmenter:
            total_samples = len(manager.file_list)
            self.manager = manager
            if n_aug:
                self.manager = SampleAugmenter(self.manager, angle_chance=0, n_aug=n_aug)
        else:
            total_samples = len(manager.manager.file_list)
            self.manager = manager
        self.preprocessing_function = preprocessing_function
        self.manager.set_batch_size(batch_size)
        super().__init__(total_samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        shape = self.input_shape
        batch_size = len(index_array)
        # Get random angles to rotate images
        thetas = self._get_thetas(batch_size)
        # Allocate arrays
        input_tensor = np.zeros((batch_size,) + shape + (self.color_channels,))
        target_tensor = thetas / self.deg_resolution
        n_classes = int(360 / self.deg_resolution)
        # Preprocessor
        if self.make_grayscale:
            preprocessor = gray_preprocessor
        else:
            preprocessor = color_preprocessor

        for i in range(batch_size):
            sample = self.manager.read_sample(index_array[i])
            sample = preprocessor(sample, thetas[i], shape, self.crop)
            input_tensor[i] = sample.get_img_arr()[:, :, :self.color_channels]
        if self.preprocessing_function:
            input_tensor = self.preprocessing_function(input_tensor)
        return input_tensor, to_categorical(target_tensor, n_classes)

    def write_sample(self, sample: Sample, write_image=False) -> int:
        pass

    def sample_generator(self, batch_size: int) -> Sample_Generator:
        pass

    def write_samples(self, samples: Sample_List, write_image=False) -> int:
        pass

    def read_sample(self, n: int) -> Sample:
        # Preprocessor
        if self.make_grayscale:
            preprocessor = gray_preprocessor
        else:
            preprocessor = color_preprocessor
        theta = self._get_thetas(1)[0]
        return preprocessor(self.manager.read_sample(n), theta, self.input_shape, self.crop)

    def _get_thetas(self, batch_size):
        # Get random angles to rotate images
        if self.rotate < 0:
            thetas = np.random.randint(0, 360, batch_size)
            # Force the angles to the correct resolution
            thetas -= np.remainder(thetas, self.deg_resolution)
        else:
            thetas = np.ones(batch_size) * self.rotate
        return thetas
