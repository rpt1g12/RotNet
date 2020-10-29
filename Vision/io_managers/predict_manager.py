from Vision import Sample
from Vision.io_managers import Manager
from Vision.pipelines.pipeline import Pipeline
from Vision.sample import Sample_Generator, Sample_List


class PredictManager(Manager):
    """Pseudo-Sample-Manager that outputs results of the KmPipeline"""

    def __init__(self, pipe: Pipeline):
        self.pipe = pipe
        manager = pipe.get_sample_manager()
        super(PredictManager, self).__init__(manager.get_in_path(), manager.get_out_path(), manager.get_file_list(),
                                             manager.batch_size, False, 0)

    def clone(self) -> 'PredictManager':
        return PredictManager(self.pipe)

    def set_batch_size(self, batch_size: int):
        self.pipe.get_sample_manager().set_batch_size(batch_size)

    def read_sample(self, n: int) -> Sample:
        sample = self.pipe.get_sample_manager().read_sample(n)
        return self.pipe.predict_sample(sample)

    def write_sample(self, sample: Sample, write_image=False) -> int:
        return self.pipe.get_sample_manager().write_sample(sample)

    def sample_generator(self, batch_size: int = 0) -> Sample_Generator:
        if batch_size > 0:
            self.set_batch_size(batch_size)
        return self.pipe.predict_batch(self.batch_size)

    def write_samples(self, samples: Sample_List, write_image=False) -> int:
        return 0

    def _get_batches_of_transformed_samples(self, index_array) -> Sample_List:
        manager = self.pipe.get_sample_manager()
        samples = manager._get_batches_of_transformed_samples(index_array)
        new_samples = list()
        for sample in samples:
            new_samples.append(self.pipe.predict_sample(sample))
        return new_samples
