# WeatherNext 2 inference profiling on GPU

## Problem statement (draft, will sharpen as we learn)
WeatherNext 2 (Google DeepMind, open-sourced Aug 2026) is optimized for TPU.
Its README says the attention implementation has to be swapped to run on GPU,
and the full 0.25° model needs an H100 for memory. Question: where does GPU
inference time and memory actually go, and how far can we hill-climb from the
stock GPU path?

## Baselines
- **B0** – official JAX code (`weathernext@v0.3.0`) on GPU, using its GPU attention fallback
- **B1** – community PyTorch port (HF Transformers branch), after numerical parity with B0
- Optimizations are measured against B1 (and compared to B0)

## Phases
1. **Setup + parity + baseline profile** (Mini model, 1°, cheap GPU)
2. Sparse neighborhood attention: node ordering + block-sparse kernel
3. Ensemble throughput (batching members, lower precision)
4. Stretch: Trainium / NKI port of the attention kernel

## Layout
- `scripts/`   – runnable code (one script per experiment)
- `notes/`     – learning log + concept explanations for the article
- `results/`   – small CSVs/plots (profiler traces are git-ignored)
- `data/`      – sample inputs + weights (git-ignored)

## Environments
Two separate virtualenvs on purpose (JAX and PyTorch ship different CUDA/cuDNN
wheels that can clash): `requirements-jax.txt`, `requirements-torch.txt`.
