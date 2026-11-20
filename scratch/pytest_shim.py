"""Minimal pytest-compatible runner for this repo's plain-function test suite."""
import importlib.util
import inspect
import json
import re
import sys
import types
from pathlib import Path


class Approx:
    def __init__(self, expected, rel=1e-9, abs=None):
        self.expected = expected
        self.rel = rel
        self.abs = 1e-12 if abs is None else abs

    def __eq__(self, other):
        if isinstance(self.expected, dict):
            return set(other) == set(self.expected) and all(
                other[key] == Approx(value, rel=self.rel, abs=self.abs)
                if isinstance(value, (int, float)) else other[key] == value
                for key, value in self.expected.items())
        if isinstance(self.expected, (list, tuple)):
            return len(other) == len(self.expected) and all(
                abs(a - b) <= max(self.rel * abs(b), self.abs)
                for a, b in zip(other, self.expected))
        return abs(other - self.expected) <= max(self.rel * abs(self.expected), self.abs)


class Raises:
    def __init__(self, exc, match=None):
        self.exc = exc
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"{self.exc.__name__} not raised")
        if not issubclass(exc_type, self.exc):
            return False
        if self.match is not None and re.search(self.match, str(exc)) is None:
            raise AssertionError(f"{exc!s} does not contain {self.match!r}")
        self.value = exc
        return True


class FixtureMarker:
    def __call__(self, func=None):
        if func is None:
            return lambda f: f
        func._is_fixture = True
        return func


class ParametrizeMarker:
    def __call__(self, argnames, cases):
        names = [n.strip() for n in argnames.split(",")]

        def decorator(func):
            wrapped_cases = []
            for case in cases:
                if len(names) == 1 and not isinstance(case, tuple):
                    wrapped_cases.append({names[0]: case})
                else:
                    wrapped_cases.append(dict(zip(names, case)))
            func._param_cases = wrapped_cases
            return func

        return decorator


pytest = types.ModuleType("pytest")
pytest.approx = lambda expected, **kw: Approx(expected, **kw)
pytest.raises = lambda exc, match=None: Raises(exc, match)
pytest.mark = types.SimpleNamespace(parametrize=ParametrizeMarker())
pytest.fixture = FixtureMarker()
sys.modules["pytest"] = pytest

sys.path.insert(0, "/workspace/src")


def make_sample_fixture():
    root = Path("/workspace/examples")

    def load(name):
        return json.loads((root / (name + ".json")).read_text())

    return load


def run_file(path, shared_fixtures):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixtures = dict(shared_fixtures)
    fixtures.update({name: obj for name, obj in vars(module).items()
                     if callable(obj) and getattr(obj, "_is_fixture", False)})
    failures, count = [], 0
    for name, func in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(func):
            continue
        params = getattr(func, "_param_cases", None)
        cases = params if params else [{}]
        for case in cases:
            kwargs = dict(case)
            for fixture_name in inspect.signature(func).parameters:
                if fixture_name in fixtures:
                    kwargs[fixture_name] = fixtures[fixture_name]()
            count += 1
            label = f"{name}[{case}]" if params else name
            try:
                func(**kwargs)
                print(f"PASS {label}")
            except Exception as exc:  # noqa: BLE001
                failures.append((label, repr(exc)))
                print(f"FAIL {label} :: {exc!r}")
    return count, failures


total, all_failures = 0, []
conftest_path = Path("/workspace/tests/conftest.py")
conftest_spec = importlib.util.spec_from_file_location("conftest", conftest_path)
conftest = importlib.util.module_from_spec(conftest_spec)
conftest_spec.loader.exec_module(conftest)
shared_fixtures = {name: obj for name, obj in vars(conftest).items()
                   if callable(obj) and getattr(obj, "_is_fixture", False)}
for test_file in sorted(Path("/workspace/tests").glob("test_*.py")):
    count, failures = run_file(test_file, shared_fixtures)
    total += count
    all_failures.extend(failures)

print()
if all_failures:
    print(f"{len(all_failures)} FAILURES out of {total}")
    for label, detail in all_failures:
        print(" -", label, detail)
    sys.exit(1)
print(f"ALL {total} TESTS PASSED")
