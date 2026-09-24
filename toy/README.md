# Toy example: RHSO on two moons

A 2-D illustration of Receding-Horizon State Optimization with a MeanFlow-style prior
(one learned jump from any time `t` to any time `s`), independent of the image benchmark.

```bash
python toy/rhso_two_moons.py                      # trains a small MLP on the CPU, then draws
python toy/rhso_two_moons.py --N 1 2 4 8 --M 40   # stages N and inner steps M per stage
python toy/rhso_two_moons.py --budget 160         # fixed total budget N * M
```

Outputs `figures/toy_rhso_two_moons.pdf` (trajectories and per-stage terminal error) and a
sanity plot of the learned transport.  The trained model is cached under `cache/`.
Only `numpy`, `torch` and `matplotlib` are required.
