"""Minimal pytest-compatible shim (development only; the project targets real pytest).

Supports the subset used by tests/: raises, approx, mark.parametrize and the
sample fixture convention through a small runner in ``run_tests.py``.
"""
import math


class Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other):
        if isinstance(self.expected, (list, tuple)):
            return len(other) == len(self.expected) and all(
                _close(a, b, self.rel, self.abs) for a, b in zip(self.expected, other))
        if isinstance(self.expected, dict):
            return set(other) == set(self.expected) and all(
                _close(self.expected[k], v, self.rel, self.abs) for k, v in other.items())
        return _close(self.expected, other, self.rel, self.abs)

    def __repr__(self):
        return f"approx({self.expected!r})"


def _close(expected, actual, rel, abs_tol):
    if expected is None:
        return actual is None
    return abs(expected - actual) <= max(abs_tol, rel * max(abs(expected), abs(actual)))


def approx(expected, rel=1e-6, abs=1e-12):
    return Approx(expected, rel, abs)


class _Mark:
    def parametrize(self, argnames, argvalues):
        names = [name.strip() for name in argnames.split(",")]

        def decorator(function):
            function._parametrize = (names, list(argvalues))
            return function

        return decorator


class MarkProxy:
    parametrize = _Mark().parametrize


mark = MarkProxy()


class raises:
    def __init__(self, exception, match=None):
        self.exception = exception
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            raise AssertionError(f"{self.exception.__name__} was not raised")
        if not issubclass(exc_type, self.exception):
            return False
        if self.match is not None:
            code = getattr(exc, "code", None)
            haystack = str(exc) if code is None else f"{code}: {exc}"
            if self.match not in haystack:
                raise AssertionError(f"pattern {self.match!r} not found in {haystack}")
        return True
