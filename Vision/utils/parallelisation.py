import threading
from concurrent.futures import Future
import concurrent.futures
from tqdm import tqdm
from typing import Callable, Iterable, List, Generator


class ThreadSafeGen:
    def __init__(self, gen):
        """
        Takes an iterator/generator and makes gen thread-safe by
        serializing call to the `next` method of given iterator/generator.

        :param gen: A Generator
        """
        self.gen = gen
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.gen.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes gen thread-safe.

    :param f: A generator
    :return: Thread-safe generator
    """

    def g(*a, **kw):
        return ThreadSafeGen(f(*a, **kw))

    return g


@threadsafe_generator
def batch_generator(iterable: Iterable, batch_size: int) -> Generator:
    """
    Returns a basic batch generator from an iterable
    :param iterable: Iterable to create the batch generator from
    :param batch_size: Size of the batches
    :return: Batch generator
    """
    batch = list()
    for element in iterable:
        batch.append(element)
        if len(batch) % batch_size == 0:
            out_batch = batch.copy()
            batch.clear()
            yield out_batch
    yield batch


def parallelize_with_thread_pool(f: Callable, iterable: Iterable, size: int = None) -> List[Future]:
    """
    Parallelises de execution of the incomming function over the iterable using threadding.
    :param f: Function to execute in parallel.
    :param iterable: Iterable to map the function to.
    :param size: Size of the iterable (Optional). Defaults to None
    :return: List of Futures
    """
    with tqdm(total=size) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(f, batch) for batch in iterable]
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
    return futures


# def parallelize_with_process_pool(f: Callable, iterable: Iterable, size: int = None) -> List[Future]:
#     """
#     Parallelises de execution of the incomming function over the iterable using multi-processing.
#     :param f: Function to execute in parallel.
#     :param iterable: Iterable to map the function to.
#     :param size: Size of the iterable (Optional). Defaults to None
#     :return: List of Futures
#     """
#     with tqdm(total=size) as pbar:
#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             futures = [executor.submit(f, batch) for batch in iterable]
#             for _ in concurrent.futures.as_completed(futures):
#                 pbar.update(1)
#     return futures
