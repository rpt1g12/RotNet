import os
from typing import List

import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa

from Vision import Sample
from Vision.io_managers import Manager
from Vision.sample import Sample_List, Sample_Generator
from Vision.utils.parallelisation import threadsafe_generator

DEFAULT_SEQ = iaa.Sequential([
    iaa.SomeOf(
        (0, 3),
        [
            iaa.Sometimes(
                0.5,
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
            ),
            iaa.Sometimes(
                0.5,
                # Strengthen or weaken the contrast in each image.
                iaa.LinearContrast((0.75, 1.5))
            ),
            # Blur each image with varying strength using
            # gaussian blur (sigma between 0 and 3.0),
            # average/uniform blur (kernel size between 2x2 and 7x7)
            # median blur (kernel size between 3x3 and 11x11).
            iaa.Sometimes(
                0.4,
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)),
                    iaa.AverageBlur(k=(2, 4)),
                    iaa.MedianBlur(k=(3, 5)),
                ])
            )
        ]
    ),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.Sometimes(
        0.4,
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
    ),
    iaa.Sometimes(
        0.8,
        iaa.OneOf([
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Change Color and Saturation of the images
            iaa.AddToHueAndSaturation((-60, 60))
        ])
    )
])


class SampleAugmenter(Manager):
    """
    Crea copias con pequenas modificaciones de una muestra o lista de muestras.
    """

    def __init__(self,
                 manager: Manager,
                 imgaug_seq=None,
                 angle_range: tuple = (-20, 20),
                 angle_chance: float = 0.5,
                 scale_range: tuple = (0.75, 0.75),
                 scale_chance: tuple = (0.3, 0.3),
                 n_aug=2,
                 seed: int = 0):
        np.random.seed(seed)
        ia.seed(seed)
        if imgaug_seq is None:
            imgaug_seq = DEFAULT_SEQ
        self.imgaug_seq = imgaug_seq
        self.seed = seed
        self.n_aug = n_aug
        assert (manager.batch_size % n_aug == 0) and (
                manager.batch_size > n_aug), "n_aug debe ser un factor de manager.batch_size"

        manager.set_batch_size(manager.batch_size // n_aug)
        file_list = [self._generate_aug_sample_name(f, i) for f in manager.get_file_list() for i in range(n_aug)]
        self.batch_size = manager.batch_size
        self.manager = manager
        self.angle_range = angle_range
        self.angle_chance = angle_chance
        self.scale_range = scale_range
        self.scale_chance = scale_chance
        super(SampleAugmenter, self).__init__(manager.get_in_path(), manager.get_out_path(), file_list,
                                              manager.batch_size, manager.shuffle, seed)

    def clone(self) -> 'SampleAugmenter':
        return SampleAugmenter(manager=self.manager,
                               imgaug_seq=self.imgaug_seq,
                               angle_range=self.angle_range,
                               angle_chance=self.angle_chance,
                               scale_range=self.scale_range,
                               scale_chance=self.scale_chance,
                               n_aug=self.n_aug,
                               seed=self.seed)

    def read_sample(self, n: int) -> Sample:
        sample = self._get_batches_of_transformed_samples([n])[0]
        id_aug = n % self.n_aug
        return self.augment_sample(sample, id_aug)

    def write_sample(self, sample: Sample, write_image=False) -> None:
        self.manager.write_sample(sample, write_image)

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.manager.set_batch_size(batch_size)

    @threadsafe_generator
    def sample_generator(self, batch_size: int = 0) -> Sample_Generator:
        assert (batch_size % self.n_aug == 0) and (
                batch_size > self.n_aug), "batch_size debe ser multiplo de n_aug"

        if batch_size > 0:
            self.set_batch_size(batch_size)

        for samples in self.manager.sample_generator(self.batch_size // self.n_aug):
            new_samples = list()
            for sample in samples:
                for id_aug in range(self.n_aug):
                    new_samples.append(self.augment_sample(sample, id_aug))
            yield new_samples

    def write_samples(self, samples: Sample_List, write_image=False) -> None:
        self.manager.write_samples(samples, write_image)

    def _generate_aug_sample_name(self, sample_name: str, id_aug:int) -> str:
        basename, extension = os.path.splitext(os.path.basename(sample_name))
        return f"{basename}_aug{id_aug:03d}{extension}"


    def augment_sample(self, sample: Sample, id_aug:int) -> Sample:
        seq = self.imgaug_seq
        angle_chance = self.angle_chance
        angle_0, angle_1 = self.angle_range
        scale_x_chance, scale_y_chance = self.scale_chance
        scale_range_x, scale_range_y = self.scale_range
        img_arr = sample.get_img_arr()
        new_img = seq(image=img_arr)
        sample = sample.substitute_img(new_img)
        sample.name = self._generate_aug_sample_name(sample.name, id_aug)
        angle = (np.random.rand() <= angle_chance) * np.random.uniform(angle_0, angle_1)
        scale_x = np.random.uniform(scale_range_x, 1) if (np.random.rand() <= scale_x_chance) else 1
        scale_y = np.random.uniform(scale_range_y, 1) if (np.random.rand() <= scale_y_chance) else 1
        return sample.affine(angle, (scale_x, scale_y))

    def _get_batches_of_transformed_samples(self, index_array) -> Sample_List:
        original_index_array = list(set([idx//self.n_aug for idx in index_array]))
        samples = self.manager._get_batches_of_transformed_samples(np.asarray(original_index_array))
        new_samples = list()
        for sample in samples:
            for id_aug in range(self.n_aug):
                new_samples.append(self.augment_sample(sample, id_aug))
        return new_samples
