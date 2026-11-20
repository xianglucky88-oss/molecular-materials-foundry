"""Tiny test runner that loads the repository tests against the pytest shim."""
import importlib.util
import inspect
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dev_scripts" / "pytest_shim"))
sys.path.insert(0, str(ROOT / "src"))
import pytest  # noqa: E402

EXAMPLES = ROOT / "examples"


def sample_fixture(name):
    return json.loads((EXAMPLES / (name + ".json")).read_text())


def collect(module):
    for name, function in vars(module).items():
        if not name.startswith("test_") or not callable(function):
            continue
        parametrize = getattr(function, "_parametrize", None)
        if parametrize is None:
            yield name, function, {}
            continue
        argnames, argvalues = parametrize
        for values in argvalues:
            if not isinstance(values, tuple):
                values = (values,)
            label = ",".join(str(v) for v in values)
            yield f"{name}[{label}]", function, dict(zip(argnames, values))


def main():
    failures = []
    passed = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for label, function, kwargs in collect(module):
            args = {}
            if "sample" in inspect.signature(function).parameters:
                args["sample"] = sample_fixture
            try:
                function(**args, **kwargs)
            except Exception:  # noqa: BLE001
                failures.append((label, traceback.format_exc()))
            else:
                passed += 1
    for label, trace in failures:
        print("FAIL", label)
        print(trace)
    print(f"{passed} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
