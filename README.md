# ML Bioprocess — Yee Collins Research Group

Computational pipeline for microbial consortium optimization.

## Pipeline stages
- **Stage 1+2** (`stage_1_2/gem_adder.py`) — GEM loading and community assembly
- **Stage 3** (`stage_3/fba_screen.py`) — FBA-based pair screening with cooperative tradeoff
- **Stage 4** — Differentiable JAX/Diffrax dynamic simulator (Wilson)

## Setup
```bash
conda create -n bioprocess python=3.11
conda activate bioprocess
pip install -r requirements.txt
```

## Quickstart
```python
from stage_1_2.gem_adder import gem_adder
from stage_3.fba_screen import fba_screen

cm = gem_adder("textbook", "salmonella")
result = fba_screen(cm)
```
