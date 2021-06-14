from typing import Tuple

import cv2
import numpy as np
from keras.utils import to_categorical

from Vision import Sample
from Vision.io_managers import Manager, SampleAugmenter
from Vision.sample import Sample_List, Sample_Generator


def gray_preprocessor(sample: Sample, theta: float, input_shape, crop) -> Sample:
    sample.set_img(cv2.cvtColor(sample.get_img_arr(), cv2.COLOR_BGR2GRAY))
    return sample.set_rotation(theta).apply_rotation(crop).resize(input_shape).set_rotation(theta)


def color_preprocessor(sample: Sample, theta: float, input_shape, crop) -> Sample:
    return sample.set_rotation(theta).apply_rotation(crop).resize(input_shape).set_rotation(theta)


def random_theta_batch(batch_size, deg_resolution):
    # Get random angles to rotate images
    thetas = np.random.randint(0, 360, batch_size)
    # Force the angles to the correct resolution
    thetas -= np.remainder(thetas, deg_resolution)
    return thetas


class RotNetManager(Manager):
    """Docstring"""

    def __init__(self,
                 manager: Manager,
                 input_shape: Tuple[int, int],
                 deg_resolution: int = 2,
                 batch_size=32,
                 rotate=-1,
                 fixed=False,
                 preprocessing_function=None,
                 crop=True,
                 n_aug=0,
                 regression=False,
                 make_grayscale: bool = False,
                 shuffle=True,
                 seed=0):
        self.input_shape = input_shape
        assert 360 % deg_resolution == 0, "Resolution must be a factor of 360"
        self.deg_resolution = deg_resolution
        self.batch_size = batch_size
        self.color_channels = 1 if make_grayscale else 3
        self.crop = crop
        if rotate >= 0:
            assert rotate % deg_resolution == 0, "Resolution must be a factor of rotate"
        self.rotate = rotate
        self.regression = regression
        self.make_grayscale = make_grayscale
        if type(manager) != SampleAugmenter:
            self.manager = manager
            if n_aug:
                self.manager = SampleAugmenter(self.manager, angle_chance=0, n_aug=n_aug)
        else:
            self.manager = manager
        self.preprocessing_function = preprocessing_function
        self.manager.set_batch_size(batch_size)
        if fixed and rotate < 0:
            self.stored_thetas = random_theta_batch(len(manager.get_file_list()), deg_resolution)
        elif rotate >= 0:
            self.stored_thetas = np.ones(len(manager.get_file_list())) * rotate
            print("Rotation is fixed because rotate>0")
            fixed = True
        else:
            self.stored_thetas = None
            fixed = False
        self.fixed = fixed
        super(RotNetManager, self).__init__(self.manager.get_in_path(), self.manager.get_out_path(),
                                            self.manager.get_file_list(),
                                            batch_size,
                                            shuffle, seed)

    def clone(self) -> "RotNetManager":
        return RotNetManager(
            manager=self.manager,
            input_shape=self.input_shape,
            deg_resolution=self.deg_resolution,
            batch_size=self.batch_size,
            rotate=self.rotate,
            fixed=self.fixed,
            regression=self.regression,
            make_grayscale=self.make_grayscale,
            shuffle=self.shuffle,
            seed=self.seed
        )

    def _get_batches_of_transformed_samples(self, index_array):
        shape = self.input_shape
        batch_size = len(index_array)
        # Get random angles to rotate images
        thetas = self._get_thetas(index_array)
        # Allocate arrays
        input_tensor = np.zeros((batch_size,) + shape + (self.color_channels,))
        if self.regression:
            target_tensor = thetas / 360
        else:
            target_tensor = thetas / self.deg_resolution
        n_classes = int(360 / self.deg_resolution)
        # Preprocessor
        if self.make_grayscale:
            preprocessor = gray_preprocessor
        else:
            preprocessor = color_preprocessor

        samples = self.manager._get_batches_of_transformed_samples(index_array)
        for i in range(batch_size):
            sample = samples[i]
            sample = preprocessor(sample, thetas[i], shape, self.crop)
            input_tensor[i] = sample.get_img_arr()[:, :, :self.color_channels]
        if self.preprocessing_function:
            input_tensor = self.preprocessing_function(input_tensor)
        if self.regression:
            return input_tensor, target_tensor
        else:
            return input_tensor, to_categorical(target_tensor, n_classes)

    def write_sample(self, sample: Sample, write_image=False) -> int:
        return self.manager.write_sample(sample, write_image)

    def delete_sample(self, n: int) -> int:
        return self.manager.delete_sample(n)

    def sample_generator(self, batch_size: int = 0) -> Sample_Generator:
        if batch_size > 0:
            self.set_batch_size(batch_size)
        pass

    def write_samples(self, samples: Sample_List, write_image=False) -> int:
        return self.manager.write_samples(samples, write_image=write_image)

    def read_sample(self, n: int) -> Sample:
        # Preprocessor
        if self.make_grayscale:
            preprocessor = gray_preprocessor
        else:
            preprocessor = color_preprocessor
        theta = self._get_thetas(np.asarray([n]))[0]
        return preprocessor(self.manager.read_sample(n), theta, self.input_shape, self.crop)

    def _get_thetas(self, index_array):
        if self.fixed:
            return self.stored_thetas[index_array]
        else:
            return random_theta_batch(len(index_array), self.deg_resolution)
