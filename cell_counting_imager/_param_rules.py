import functools
from typing import Iterable
import inspect


def require_b_not_none_if_a(name_a: str, names_b: Iterable[str]):
    def outer(f):
        def check_arg_values(*args, **kwargs):
            param_names = list(inspect.signature(f).parameters.keys())
            kwargs.update({
                param_names[i]: val for (i, val) in enumerate(args)})
            val_a = kwargs.get(name_a)
            for name_b in (names_b if val_a else ()):
                if kwargs.get(name_b) is None:
                    raise ValueError(
                        f"{name_b} cannot be None if {name_a} is {val_a!r}")

        @functools.wraps(f)
        def inner(*args, **kwargs):
            check_arg_values(*args, **kwargs)
            return f(*args, **kwargs)
        inner.check_arg_values = check_arg_values
        return inner
    return outer
