import cv2
from typing import Tuple

import numpy as np
from Vision import Sample
from Vision.io_managers import Manager, SampleAugmenter
from Vision.sample import Sample_List, Sample_Generator
from keras.utils import to_categorical


def gray_preprocessor(sample: Sample, theta: float, input_shape, crop) -> Sample:
    sample.set_img(cv2.cvtColor(sample.get_img_arr(), cv2.COLOR_BGR2GRAY))
    return sample.set_rotation(theta).apply_rotation(crop).resize(input_shape)


def color_preprocessor(sample: Sample, theta: float, input_shape, crop) -> Sample:
    return sample.set_rotation(theta).apply_rotation(crop).resize(input_shape)


class VocRotGenerator(Manager):
    """Docstring"""

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
        thetas = np.random.randint(0, 360, batch_size)
        # Force the angles to the correct resolution
        thetas -= np.remainder(thetas, self.deg_resolution)
        return thetas

    def __init__(self, manager: Manager, input_shape: Tuple[int, int],
                 deg_resolution: int = 2, batch_size=32,
                 preprocessing_function=None,
                 crop=True,n_aug=0,
                 make_grayscale: bool = False, seed=0):
        self.input_shape = input_shape
        assert 360 % deg_resolution == 0, "Resolution must be a factor of 360"
        self.deg_resolution = deg_resolution
        self.batch_size = batch_size
        self.color_channels = 1 if make_grayscale else 3
        self.crop = crop
        self.make_grayscale = make_grayscale
        if n_aug and type(manager) != SampleAugmenter:
            self.manager = SampleAugmenter(manager,angle_chance=0,n_aug=n_aug)
        else:
            self.manager = manager
        self.preprocessing_function = preprocessing_function
        self.manager.set_batch_size(batch_size)
        np.random.seed(seed)

    def __getitem__(self, index):
        batch = self.manager[index]
        shape = self.input_shape
        batch_size = len(batch)
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
            sample = preprocessor(batch[i], thetas[i], shape, self.crop)
            input_tensor[i] = sample.get_img_arr()[:, :, :self.color_channels]
        if self.preprocessing_function:
            input_tensor = self.preprocessing_function(input_tensor)
        return input_tensor, to_categorical(target_tensor, n_classes)

    def __len__(self):
        return len(self.manager)
