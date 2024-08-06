import time
import functools
import logging

timeit_logger = logging.getLogger("timeit")


def timeit(name=None):
    def args_wrapper(func):
        @functools.wraps(func)
        def _timeit(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end = time.perf_counter() - start
                func_name = name or func.__name__
                timeit_logger.info(
                    f"Function: {func_name}, Time: {end:.2f} s"
                )

        return _timeit

    return args_wrapper
