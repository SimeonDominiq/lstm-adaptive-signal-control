#!/usr/bin/env python3
"""Verify the traffic simulation research environment."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


EXPECTED_PACKAGES = {
    "eclipse-sumo": "1.27.1",
    "sumo-data": "1.27.1",
    "traci": "1.27.1",
    "sumolib": "1.27.1",
    "tensorflow": "2.18.0",
    "numpy": "2.0.2",
    "pandas": "3.0.0",
    "matplotlib": "3.10.0",
    "scipy": "1.18.1",
    "scikit-learn": "1.9.0",
}


class Results:
    def __init__(self) -> None:
        self.passes: list[str] = []
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def passed(self, message: str) -> None:
        self.passes.append(message)
        print(f"PASS: {message}")

    def failed(self, message: str) -> None:
        self.failures.append(message)
        print(f"FAIL: {message}")

    def warned(self, message: str) -> None:
        self.warnings.append(message)
        print(f"WARN: {message}")


def run_binary(results: Results, name: str, expected_version: str) -> None:
    executable = shutil.which(name)
    if executable is None:
        results.failed(f"{name} is not on PATH")
        return

    print(f"{name} path: {executable}")
    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        results.failed(f"{name} --version could not run: {exc}")
        return

    output = (completed.stdout + completed.stderr).strip()
    print(output)
    if completed.returncode != 0:
        results.failed(f"{name} --version exited with {completed.returncode}")
    elif expected_version not in output:
        results.failed(f"{name} did not report expected version {expected_version}")
    else:
        results.passed(f"{name} reports version {expected_version}")


def main() -> int:
    results = Results()

    print("=== Host Python ===")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    if sys.version_info[:2] == (3, 12):
        results.passed("Python is version 3.12")
    else:
        results.failed("Python is not version 3.12")

    print("\n=== Installed package versions ===")
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = version(package)
            print(f"{package}: {actual}")
            if actual == expected:
                results.passed(f"{package} matches {expected}")
            else:
                results.failed(f"{package} is {actual}; expected {expected}")
        except PackageNotFoundError:
            results.failed(f"{package} is not installed")

    print("\n=== SUMO environment ===")
    sumo_home_value = os.environ.get("SUMO_HOME")
    print(f"SUMO_HOME: {sumo_home_value!r}")
    if not sumo_home_value:
        results.failed("SUMO_HOME is not set")
    else:
        sumo_home = Path(sumo_home_value)
        if not sumo_home.is_dir():
            results.failed(f"SUMO_HOME does not exist: {sumo_home}")
        elif not (sumo_home / "tools").is_dir() or not (sumo_home / "data").is_dir():
            results.failed("SUMO_HOME does not contain both tools/ and data/")
        else:
            results.passed("SUMO_HOME exists and contains tools/ and data/")

    for module_name in ("traci", "sumolib"):
        try:
            module = importlib.import_module(module_name)
            print(f"{module_name} module: {Path(module.__file__).resolve()}")
            results.passed(f"import {module_name} succeeded")
        except Exception as exc:
            results.failed(f"import {module_name} failed: {exc}")

    print("\n=== SUMO binaries ===")
    run_binary(results, "sumo", "1.27.1")
    run_binary(results, "netconvert", "1.27.1")

    print("\n=== libsumo ===")
    try:
        libsumo = importlib.import_module("libsumo")
        print(f"libsumo module: {Path(libsumo.__file__).resolve()}")
        results.passed("libsumo is importable")
    except Exception as exc:
        results.warned(f"libsumo is not importable: {exc}")

    print("\n=== TensorFlow LSTM forward pass ===")
    try:
        import tensorflow as tf

        print(f"TensorFlow: {tf.__version__}")
        physical_devices = tf.config.list_physical_devices()
        logical_devices = tf.config.list_logical_devices()
        print("Physical devices:")
        for device in physical_devices:
            print(f"  {device}")
        print("Logical devices:")
        for device in logical_devices:
            print(f"  {device}")

        tf.keras.utils.set_random_seed(20260827)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(4, 3)),
                tf.keras.layers.LSTM(4, return_sequences=True),
                tf.keras.layers.LSTM(2),
                tf.keras.layers.Dense(1),
            ]
        )
        output = model(tf.zeros((1, 4, 3), dtype=tf.float32), training=False)
        print(f"Forward-pass output shape: {tuple(output.shape)}")
        print(f"Forward-pass output: {output.numpy().tolist()}")
        if tuple(output.shape) == (1, 1):
            results.passed("TensorFlow built and ran a two-layer LSTM forward pass")
        else:
            results.failed(f"TensorFlow returned unexpected output shape {tuple(output.shape)}")
    except Exception as exc:
        results.failed(f"TensorFlow LSTM verification failed: {type(exc).__name__}: {exc}")

    print("\n=== SUMMARY ===")
    print(f"PASS checks: {len(results.passes)}")
    print(f"WARN checks: {len(results.warnings)}")
    print(f"FAIL checks: {len(results.failures)}")
    for warning in results.warnings:
        print(f"WARN: {warning}")
    for failure in results.failures:
        print(f"FAIL: {failure}")

    if results.failures:
        print("OVERALL: FAIL")
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
