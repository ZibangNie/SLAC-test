# Windows direct-dependency metadata gate

2026-10-04. All eight frozen direct-release candidates passed this metadata-only gate for Windows CPython 3.12.10. This establishes compatible wheel metadata and consistency among the selected direct pins. Transitive resolution, installation, CUDA compatibility and model execution remain untested.

The candidates came from the [Kev `kev-1.0` package manifest](https://github.com/jaredpalmer/kev/blob/6b719c3c3f367295f6ef336f4f751cf5ff970abc/pyproject.toml), fixed at source commit `6b719c3c3f367295f6ef336f4f751cf5ff970abc`. Torch 2.8.0 was selected within the declared `>=2.6,<2.9` interval; the other exact pins use normalized declared lower bounds. No version search or retries occurred. These eight packages are the declared package-level candidates, not a proven minimal inference dependency set. Whether a minimal path needs datasets or scikit-learn has not been established.

## Observed result

The run made **8 fixed-version PyPI JSON requests**, read **408,515 response-body bytes** and took **8.688 seconds**. The byte count excludes transport overhead. It checked release identity, `Requires-Python`, non-yanked wheel tags and constraints among the selected direct pins with optional extras inactive. It did not solve the transitive dependency graph.

| Fixed PyPI version | Compatible wheel listed by PyPI | Artifact size in metadata, bytes |
| --- | --- | ---: |
| [accelerate 1.15.0](https://pypi.org/project/accelerate/1.15.0/) | `accelerate-1.15.0-py3-none-any.whl` | 394,295 |
| [datasets 3.0.0](https://pypi.org/project/datasets/3.0.0/) | `datasets-3.0.0-py3-none-any.whl` | 474,265 |
| [numpy 2.5.3](https://pypi.org/project/numpy/2.5.3/) | `numpy-2.5.3-cp312-cp312-win_amd64.whl` | 12,567,828 |
| [peft 0.21.0](https://pypi.org/project/peft/0.21.0/) | `peft-0.21.0-py3-none-any.whl` | 832,883 |
| [pydantic 2.9.0](https://pypi.org/project/pydantic/2.9.0/) | `pydantic-2.9.0-py3-none-any.whl` | 434,325 |
| [scikit-learn 1.9.1](https://pypi.org/project/scikit-learn/1.9.1/) | `scikit_learn-1.9.1-cp312-cp312-win_amd64.whl` | 8,262,238 |
| [torch 2.8.0](https://pypi.org/project/torch/2.8.0/) | `torch-2.8.0-cp312-cp312-win_amd64.whl` | 241,326,087 |
| [transformers 5.17.0](https://pypi.org/project/transformers/5.17.0/) | `transformers-5.17.0-py3-none-any.whl` | 12,295,140 |

These are metadata observations. No wheel, source archive, tokenizer or model weight was downloaded. Artifact sizes above are not bytes transferred by this run, and the recorded artifact SHA256 values have not been verified against downloaded files. The Torch filename does not establish whether this is a CPU or CUDA build; that remains undecided.

There were no package installations, environment creation, upstream-code execution, GPU use or model API calls. The existing SLAC environment was unchanged. Limits were 8 requests, 512 KiB per response, 4 MiB total response bodies, 5-second socket timeout, a 50-second metadata work budget and a 60-second child-process hard timeout. The first missing version, incompatible candidate or exceeded bound would have stopped the path.

## Provenance

The [aggregate JSON](results/local_decision_metadata_20261004.json) retains the eight exact source URLs, wheel filenames, metadata sizes and SHA256 values, execution measurements and source/input/script identities. This publication was derived entirely from the existing private plan, result and execution receipts, with no additional network requests.

| Recorded item | SHA256 |
| --- | --- |
| Frozen private plan | `95698fded9076f7e79c87139c4bb03530e2d8b670de5bda7fc9c67dd4e2ef9ba` |
| Private parsed result | `487b91813e30999e14b786a2f5e8798d16bc5615fe19c91168ed4a033629b630` |
| Private execution receipt | `2a6bbb87e65d5ffbfcec10e07116bc2a5e4cab1e9035d024f3c6f51b474939a4` |
| Executed controller source, UTF-8 | `db6d234046e2e7af0e86d37fcd4fe29aa533700ce5ca7e230540c5926869cc64` |
| Executed metadata-check source, UTF-8 | `84f5831e22aeeb76e8f29e7839466c9993f3c6af115d4767a22967f995f91a64` |
| Published aggregate JSON | `f43606cff81cc1ca41968442e064b98cb1ccaa5b51d0114a5569d4550536d749` |

The execution receipt's plan/result hashes were checked against the saved bytes. Script hashes cover the exact source strings embedded in the frozen plan, extracted without execution. Raw HTTP response bodies were not retained; the parsed-result hash must not be described as a hash of the complete upstream responses. Long optional dependency lists remain outside the published aggregate.

## Next boundary

The next step is a separately bounded transitive-resolution check, followed by any import check in a fresh isolated environment. The exact Torch CPU/CUDA distribution, Windows native dependencies and GPU compatibility must be settled before a model smoke test. This pass does not justify modifying the working research environment or starting checkpoint downloads.

The result establishes no model quality, conditional-decision accuracy, structure benefit, Answer F1 gain, novelty or equivalence between Kev and JEV. The earlier [local feasibility note](LOCAL_DECISION_FEASIBILITY_20261004.md) remains an unchanged historical snapshot.
