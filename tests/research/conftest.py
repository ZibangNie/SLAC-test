"""Isolate clock assumptions in three historically sealed synthetic suites.

The two client suites already replace their runner's ``utc_now`` but the shared client
records ``started_at`` through its own imported ``datetime``.  Align that one
client clock with the test's runner clock; otherwise replay correctly rejects
synthetic ledgers created after the historical 2026-09-27 cutoff.

This client-clock fixture is limited to those two exact test paths.  It does not change the
system clock, production files, cutoff constants, monotonic timers, or replay
validation.  Dynamic delegation preserves each test's explicit deadline and
tampered-timestamp cases.  Pytest restores the client import after every test.

One exact native-index integration test additionally declares positive fake
encoding time while its fast stub can finish within a Windows monotonic tick.
Its runner receives a local monotonic proxy; the global time module and real
perf_counter measurements remain unchanged.
"""
from datetime import datetime
from pathlib import Path

import pytest


_SYNTHETIC_CLOCK_TESTS = {
    Path(__file__).with_name('test_qasper_primary_support_recovery.py').resolve(),
    Path(__file__).with_name('test_qasper_relation_content.py').resolve(),
}


@pytest.fixture(autouse=True)
def align_sealed_synthetic_client_clock(request, monkeypatch):
    if Path(request.module.__file__).resolve() not in _SYNTHETIC_CLOCK_TESTS:
        return
    runner = request.module.m

    class SyntheticClientDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            # Resolve on each call so test-local utc_now boundary overrides win.
            instant = runner.utc_now()
            if instant.utcoffset() is None:
                raise AssertionError('synthetic runner clock must be timezone-aware')
            return instant.astimezone(tz) if tz is not None else instant.astimezone().replace(tzinfo=None)

    monkeypatch.setattr(runner.client, 'datetime', SyntheticClientDatetime)


@pytest.fixture(autouse=True)
def align_native_stub_elapsed_clock(request, monkeypatch):
    target = Path(__file__).with_name('test_qasper_native_dual_index.py').resolve()
    if (Path(request.module.__file__).resolve() != target or request.node.name !=
            'test_synthetic_full_run_output_seals_and_failure_stop_no_retry'):
        return
    experiment = request.module.experiment
    real_time = experiment.time

    class SyntheticElapsedTime:
        def __init__(self):
            # Keep the real epoch because imported deadline helpers use real time.
            self.tick = real_time.monotonic()

        def monotonic(self):
            self.tick += .01
            return self.tick

        def __getattr__(self, name):
            return getattr(real_time, name)

    monkeypatch.setattr(experiment, 'time', SyntheticElapsedTime())
