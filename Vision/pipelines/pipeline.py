import abc
import time

from Vision import Sample
from Vision.io_managers import Manager
from Vision.sample import Sample_Generator


class Pipeline(metaclass=abc.ABCMeta):
    """
    Clase abstracta que representa un proceso de lectura de una muestra
    y su evaluacion por uno o varios modelos
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
                hasattr(subclass, 'predict_sample') and
                callable(subclass.predict_sample) and
                hasattr(subclass, 'get_sample_manager') and
                callable(subclass.get_sample_manager) or
                NotImplemented
        )

    @abc.abstractmethod
    def get_sample_manager(self) -> Manager:
        return NotImplementedError

    @abc.abstractmethod
    def predict_sample(self, sample: Sample) -> Sample:
        return NotImplementedError

    def predict_batch(self, batch_size: int) -> Sample_Generator:
        sample_gen = self.get_sample_manager().sample_generator(batch_size)
        for samples in sample_gen:
            preds = list()
            t0 = time.time()
            for sample in samples:
                pred = self.predict_sample(sample)
                preds.append(pred)
            t1 = time.time()
            t_batch = t1 - t0
            print("{:.2f}s per batch".format(t_batch))
            print("{:.2f}s per prediction".format(t_batch / len(preds)))
            yield preds

