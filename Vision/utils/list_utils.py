import functools
import operator


def flatten_list_of_lists(l: list) -> list:
    """
    Returns the flattened version of the multidimensional input list
    :param l: Input list
    :return: 1-D list
    """
    return functools.reduce(operator.iconcat, l, [])
