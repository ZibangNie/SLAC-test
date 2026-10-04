# Local decision-model feasibility

2026-10-04. Read-only reconnaissance while implementing the conditional JEV contract. No weights downloaded, packages installed, model inference, training or paid API calls occurred in this check.

## A possible local baseline

[Kev](https://github.com/jaredpalmer/kev) is an open-weight family of typed decision models with a TypeSafe-compatible interface. Its authors list a 0.8B option for a 4 GB GPU and publish a fixed `kev-1.0` source release. This is a potential **additional local baseline**, not the JEV model and not evidence that its decisions would transfer to this RAG task. The authors explicitly place the smaller model behind larger alternatives on generalization; no reported benchmark score is being adopted as our own result. See the [0.8B model card](https://huggingface.co/jaredpalmer/kev-0.8b).

The [4B model card](https://huggingface.co/jaredpalmer/kev-4b) reports 14.3 GB resident CUDA memory for its served configuration. That configuration exceeds this machine's measured GPU capacity; do not assume it fits merely from the nominal parameter count. A short 0.8B smoke test is the more plausible first local target. Actual Windows compatibility, memory consumption and latency remain unverified.

## Current local measurements

The read-only `nvidia-smi` query reported an RTX 5070 Laptop GPU with 8,151 MiB total and 7,879 MiB free at inspection. Physical system memory was 16,790,577,152 bytes, and D: had about 178 GB free. These are point-in-time measurements, not guarantees of available capacity during inference.

The existing research environment has `torch 2.9.1+cu130`, `transformers 4.57.3`, `safetensors 0.7.0`, and no installed `peft` or `bitsandbytes` distribution. The fixed release's [package specification](https://github.com/jaredpalmer/kev/blob/6b719c3c3f367295f6ef336f4f751cf5ff970abc/pyproject.toml) asks for Python 3.12–3.13, `torch>=2.6,<2.9`, `transformers>=5.17,<6` and `peft>=0.21`, among other packages. Its declared requirements differ materially from the working SLAC environment.

## Fixed source inspection

Six selected source files were read without executing upstream code. The [kev-1.0 release](https://github.com/jaredpalmer/kev/releases/tag/kev-1.0) resolves to source commit `6b719c3c3f367295f6ef336f4f751cf5ff970abc`. Its release identifies the 0.8B weight revision by the short hash `9a45d25e`; a complete base/adapter/head provenance chain has not yet been established from the actual artifacts.

The library [load options](https://github.com/jaredpalmer/kev/blob/6b719c3c3f367295f6ef336f4f751cf5ff970abc/kev/checkpoint.py#L75) permit `backend="torch", fused=False, cuda_graphs=False`. Custom fused/graph modules are conditionally imported. The single-input [model.probs path](https://github.com/jaredpalmer/kev/blob/6b719c3c3f367295f6ef336f4f751cf5ff970abc/kev/model.py#L358) can use ordinary Torch. The server defaults and the batch path introduce additional graph behavior and are unnecessary for a first smoke test. Transformers' [Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L227) includes ordinary Torch fallback functions, but its optional kernel dispatcher and actual Windows execution remain unverified. Source availability is not a successful compatibility test.

The expected `head.pt` is a metadata dictionary plus a two-projection state dict (`q.weight`, `q.bias`, `k.weight`, `k.bias`), including base revision and temperature. No head file was downloaded or loaded. Future inspection should explicitly use `torch.load(..., map_location="cpu", weights_only=True)` and stop on incompatible contents rather than enabling unrestricted deserialization; see the [loader source](https://github.com/jaredpalmer/kev/blob/6b719c3c3f367295f6ef336f4f751cf5ff970abc/kev/checkpoint.py#L41) and [PyTorch serialization documentation](https://docs.pytorch.org/docs/stable/notes/serialization.html).

## Next feasibility gate

The next gate is isolated Windows dependency resolution, followed by imports with network access disabled and a tiny randomly initialized forward check with fused kernels and CUDA graphs explicitly disabled. Preserve the current SLAC environment. Only after that gate passes should the exact base/adapter revisions, licenses and weight formats be fixed and a checkpoint considered for one short, invented state under a strict local runtime/memory limit, before the fixed synthetic pairs. No corpus sweep, fine-tuning, cloud GPU or paid hosted Kev endpoint is warranted by this reconnaissance.

A successful smoke test would establish only local executability. A subsequent fixed synthetic readout could establish behavior of **that local checkpoint** on authored cases. It would neither validate JEV nor establish source-structure benefit or new Answer F1. If platform dependencies or memory prevent the small local path, retain that measured limitation and consider a separately bounded small JEV contract probe under the user's existing small-scale exploration authorization; do not silently expand the model, dataset or budget.
