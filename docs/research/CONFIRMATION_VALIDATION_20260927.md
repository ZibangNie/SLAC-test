# Confirmation preparation: validation record

This records the checks performed while preparing the prospective metadata cohort and V2 analysis on 27 September 2026. These are software and provenance checks, not additional quality experiments or independent scientific samples.

- Metadata policy: 69 synthetic tests cover complete ledgers, identity/version matching, transitive quarantine, unknown-source review, original-family preservation and fixed aggregate privacy.
- Metadata adapter: 73 synthetic tests cover source joins, full denominators, exact seals, policy-free preparation, one-use registration, failures, tampering after resealing and complete replay. A separate metadata-only source projection checks all 20 direct real bindings before policy execution.
- Statistics V1: 83 synthetic tests. Independent review subsequently found a missing floating-point degeneracy case; this version is retained as history and is superseded by V2 for later analysis.
- Statistics V2: 90 synthetic tests, independently rerun. They preserve the V1 coverage and add the decimal constant-gain counterexample, inclusive numerical threshold, real variation above the threshold, actual SD retention and explicit V2 schema. Student t quantiles are checked against closed forms, a known quantile and an independent density integral; unequal-family bootstrap results are compared with explicit expanded draws.
- The standalone metadata verifier has 11 synthetic tests. Its real run reconstructs the graph with breadth-first traversal instead of the formal implementation's union-find, verifies 31 bound files and matches the complete ledger, proposal, components and public output bytes. This checks policy implementation rather than certifying dataset independence.

## Historical synthetic-clock repair

The first complete research-suite pass reported 1,478 passes and 25 failures in two previously sealed test modules. Their runner clocks were already mocked, but the shared request client stamped synthetic ledgers with the actual date. After the real overnight deadline, replay correctly rejected those synthetic timestamps. The added `tests/research/conftest.py` aligns only those two test modules' client clocks with their existing per-test runner clocks. It leaves production code, historical tests, plans, cutoff constants and timestamp validation unchanged, and restores the patch after every test. Explicit boundary overrides still apply.

A subsequent suite pass reported 1,592 passes and one failure in the existing native dual-index synthetic complete-run test. Its saved synthetic report had encoding duration `0.00001` seconds but total run duration `0.0`, with valid memory counters. That fixture mixed a fixed positive encoding stub with the operating system's coarse monotonic clock. A second fixture targets only that exact test node and gives its runner a private monotonic-time proxy advancing by 0.01 seconds; other time functions remain delegated to the real module. It does not change Python's global clock or weaken the audit. The six relevant test files then passed all 315 tests. Historical production files and tests remain byte-identical. No real experiment was restarted to fix a test.

## Final regression outcome

`python -m pytest tests/research -q --tb=short` completed with **1,593 passed**, two existing SWIG deprecation warnings, in 64.96 seconds. This includes all 315 new policy/adapter/V1/V2 synthetic cases and the existing research suite. The independently executed metadata verifier's 11 tests are separate. The final test harness SHA-256 is `649404b954eb95484e2cc811e153bc141364945813bb02f82fef75afce9b644a`.

## Reproducibility scope

Only synthetic new-cohort statistics have been analyzed. Existing regression tests may replay their already exposed, frozen development fixtures; they do not supply new-cohort outcomes. Exact source hashes and the public protocol bind the V2 analysis and metadata procedure. The complete local metadata receipts remain in ignored artifacts. Model-training exposure, external use and sampling assumptions are separate research limitations and are not settled by passing tests.
