import concurrent.futures
from typing import Callable

from tqdm import tqdm

from Vision.io_managers import Manager
from Vision.sample import Sample_List, Sample


class Translator(object):
    """Translates from one man_in to another"""

    def __init__(self, from_man: Manager, to_man: Manager, batch_size=None,
                 predicate: Callable[[Sample], bool] = lambda s: True):
        super(Translator, self).__init__()
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = from_man.batch_size
        self.from_man = from_man
        self.to_man = to_man
        self.to_man.set_batch_size(self.batch_size)
        self.predicate = predicate

    def __call__(self, write_image=False, *args, **kwargs):
        self.translate(write_image=write_image)

    def translate(self, write_image=False):
        from_man = self.from_man
        to_man = self.to_man
        batch_size = self.batch_size
        from_man.set_batch_size(batch_size)
        to_man.set_batch_size(batch_size)
        predicate = self.predicate

        def write_function(samples: Sample_List) -> int:
            return to_man.write_samples(samples, write_image)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            with tqdm(total=len(from_man)) as progress:
                futures = []
                for batch in from_man.sample_generator(batch_size):
                    future = executor.submit(write_function, filter(predicate, batch))
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)
