import cv2
import numpy as np

from Vision import Sample
from Vision.io_managers import Manager
from Vision.sample import Sample_List, Sample_Generator
from Vision.utils.parallelisation import threadsafe_generator


class PreProcess(Manager):
    """
    Modifica una muestra o lista de muestras en pre-proceso para su prediccion
    """

    def __init__(self,
                 manager: Manager,
                 img_shape: tuple = (1, 1),
                 make_grayscale: bool = False,
                 seed: int = 0):
        np.random.seed(seed)
        self.batch_size = manager.batch_size
        self.manager = manager
        self.img_shape = img_shape
        self.make_grayscale = make_grayscale

        super(PreProcess, self).__init__(manager.get_in_path(), manager.get_out_path(), manager.get_file_list(),
                                         manager.batch_size, manager.shuffle, seed)

    def clone(self) -> 'PreProcess':
        return PreProcess(
            manager=self.manager,
            img_shape=self.img_shape,
            make_grayscale=self.make_grayscale,
            seed=self.seed
        )

    def read_sample(self, n: int) -> Sample:
        sample = self.manager.read_sample(n)
        return self.preprocess_sample(sample)

    def write_sample(self, sample: Sample, write_image=False) -> None:
        self.manager.write_sample(sample, write_image)

    def write_samples(self, samples: Sample_List, write_image=False) -> None:
        self.manager.write_samples(samples, write_image)

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.manager.set_batch_size(batch_size)

    @threadsafe_generator
    def sample_generator(self, batch_size: int = 0) -> Sample_Generator:
        if batch_size > 0:
            self.set_batch_size(batch_size)
        for samples in self.manager.sample_generator(self.batch_size):
            new_samples = list()
            for sample in samples:
                new_samples.append(self.preprocess_sample(sample))
            yield new_samples

    def preprocess_sample(self, sample: Sample) -> Sample:

        if self.make_grayscale is True:
            sample.set_img(cv2.cvtColor(sample.get_img_arr(), cv2.COLOR_BGR2GRAY))

        if self.img_shape != sample.get_img_arr().shape[:-1]:
            sample.resize(self.img_shape)

        return sample

    def preprocess_batch(self, batch: Sample_List) -> Sample_List:
        return [self.preprocess_sample(sample) for sample in batch]

    def _get_batches_of_transformed_samples(self, index_array) -> Sample_List:
        raw_batch = self.manager._get_batches_of_transformed_samples(index_array)
        return self.preprocess_batch(raw_batch)
