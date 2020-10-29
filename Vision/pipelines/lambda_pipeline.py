from typing import Callable

from Vision import Sample
from Vision.io_managers import Manager
from Vision.pipelines.pipeline import Pipeline


class LambdaPipe(Pipeline):
    """Simple Pipeline that applies a function to every sample."""

    def __init__(self, manager: Manager, lambda_fn: Callable[[Sample], Sample]):
        super(LambdaPipe, self).__init__()
        self.manager = manager
        self.lambda_fn = lambda_fn

    def get_sample_manager(self) -> Manager:
        return self.manager

    def predict_sample(self, sample: Sample) -> Sample:
        return self.lambda_fn(sample)
