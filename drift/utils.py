"""
Some utils
"""
import time


def timeit(info):
    def decorator(func):
        def wraped_func(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            print("{}: {:.4f} sec".format(info, time.time() - start))
            return res
        return wraped_func
    return decorator
