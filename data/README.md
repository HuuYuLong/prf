# Data directory

Place all datasets here (this is the default data path used by `src/dataloaders/base.py`).

- **MNIST / CIFAR-10**: downloaded automatically by `torchvision` on first run.
- **LRA datasets**: download from the original [Long Range Arena benchmark](https://github.com/google-research/long-range-arena)
  and place them here, e.g. `data/listops/`, `data/imdb/`, `data/aan/`, `data/pathfinder/` (PathX is a larger
  Pathfinder variant from the same benchmark).
- `getdata.sh` only downloads language-modeling corpora (WikiText-2/103, enwik8, text8, PTB) and is **not**
  required for the PRF paper experiments.

Raw dataset files are intentionally **not** included in this release.
