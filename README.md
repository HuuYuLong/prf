# PRF: Parallel Resonate and Fire Neuron for Long Sequence Learning in Spiking Neural Networks

Official code release for the paper **"PRF: Parallel Resonate and Fire Neuron for Long Sequence Learning in Spiking Neural Networks"**.

The release contains the **PRF neuron**, the **SD-TCM (Spike-Driven Temporal and Channel Mixer)** backbone, and the experiment configurations used in the paper for:

- **Long Range Arena (LRA)**: ListOps, IMDB (Text), Retrieval (AAN), Image (sequential CIFAR), Pathfinder, PathX
- **Sequential image tasks**: sMNIST, psMNIST, sequential CIFAR
- **Efficiency / energy analysis**: firing-rate logging and neuron-level dynamic energy estimation

The PRF neuron and the SD-TCM/SpikingBlock layers are implemented in
`src/models/sequence/modules/spikingblock.py` (classes `PRFNeuron`, `PRFNeuronReset`, `PRFNeuronTrain`, `DecoupledResetLIF`, `SpikingBlockGate`, etc.),
with baseline neurons (BHRF, ELM, TC-LIF, TS-LIF, ParaLIF, PSN variants) under
`src/models/sequence/modules/`.

## Installation

**Option A — conda (recommended).** `environment.yml` creates an env named `prf` with the
exact package versions used to run the paper's experiments (Python 3.10, torch 2.2.2+cu118):

```bash
conda env create -f environment.yml
conda activate prf
```

**Option B — pip.** The same versions are pinned in `requirements.txt`:

```bash
conda create -n prf python=3.10 -y
conda activate prf
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 torchtext==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu118   # match your CUDA version
pip install -r requirements.txt
```

Main dependencies: PyTorch 2.2.2, PyTorch Lightning 2.5.2, Hydra 1.3.2, SpikingJelly 0.0.0.0.14, einops, timm, wandb.
A few optional baselines / visualization utilities require extra packages (see the comments in `requirements.txt`); they are **not** needed for the main PRF experiments.

## Data preparation

All datasets are expected under `./data/` (this is the default resolved by `src/dataloaders/base.py`).

- **MNIST / CIFAR-10**: downloaded automatically by `torchvision` on first run.
- **LRA datasets** (ListOps, IMDB, AAN/Retrieval, Pathfinder, PathX): download from the original
  [Long Range Arena benchmark](https://github.com/google-research/long-range-arena) and place them under `./data/`
  (e.g. `data/listops/`, `data/imdb/`, `data/aan/`, `data/pathfinder/`).
- `data/getdata.sh` only fetches language-modeling corpora (WT2/WT103/enwik8/text8/PTB) and is **not**
  required for the experiments in this paper.

## Reproducing the paper

Run everything from the repository root (Hydra resolves `configs/` relative to `train.py`).
Add `wandb=null` to disable W&B logging, and `loader.num_workers=0` if you run on Windows / a single GPU box.
Overrides follow Hydra syntax: `key=value` modifies an existing config entry, while `+key=value`
adds a new one (e.g. `model.layer.spatial`/`model.layer.temporal`/`model.layer.tau` are not defined
in the default `snn-cifar`/`snn-mnist` configs, so they need the `+` prefix there).

### LRA benchmark (Table 4)

```bash
# ListOps (T=2048)
python -m train experiment=lra/snn-listops wandb=null loader.num_workers=0 \
    model.layer.fr_scale=12 model.layer.spatial='Binary' train.monitor=test/accuracy

# IMDB / Text (T=4096)
python -m train experiment=lra/snn-imdb wandb=null loader.num_workers=0 \
    model.layer.fr_scale=16 model.layer.spatial='Binary' train.monitor=test/accuracy

# Retrieval / AAN (T=4000)
python -m train experiment=lra/snn-aan wandb=null loader.num_workers=0 \
    model.layer.fr_scale=14 model.layer.spatial='Binary' train.monitor=test/accuracy

# Image / sequential CIFAR (T=1024)
python -m train experiment=lra/snn-cifar wandb=null model.layer.fr_scale=0.25 optimizer.lr=0.005 \
    train.monitor=test/accuracy dataset.augment=true dataset.cutout=true loader.batch_size=50 \
    model.layer.bidirectional=true +model.layer.spatial='Binary'

# Pathfinder (T=1024)
python -m train experiment=lra/snn-pathfinder wandb=null loader.num_workers=0 \
    model.layer.fr_scale=0.6 model.layer.bidirectional=true model.layer.spatial='Binary' \
    train.monitor=test/accuracy encoder=embedding

# PathX (T=16384)
python -m train experiment=lra/snn-pathx wandb=null loader.num_workers=0 \
    model.layer.tau=100. model.layer.fr_scale=10 model.layer.spatial='Binary' \
    model.layer.bidirectional=true model.dropout=0. \
    optimizer.lr=0.001 optimizer.weight_decay=0.2 loader.batch_size=32 \
    train.optimizer_param_grouping.bias_weight_decay=true \
    train.optimizer_param_grouping.normalization_weight_decay=true \
    train.monitor=test/accuracy encoder=embedding
```

**Multi-seed runs (mean ± std):** append `train.seed=<seed>` to any command above, e.g.
`train.seed=0`, `train.seed=1`, `train.seed=2`, and average the reported `test/accuracy`.

> Note: the commands below are the best tuned settings from our experiments; a few of them
> (e.g. psMNIST with `tau=5`) differ slightly from the configurations reported in the paper,
> which were obtained under an earlier tuning protocol.

### sMNIST / psMNIST / seqCIFAR small models (Table 3)

```bash
# sMNIST
python -m train experiment=lra/snn-mnist wandb=null dataset.permute=false loader.num_workers=0 \
    train.monitor=test/accuracy +model.layer.tau=2. model.n_layers=4 model.d_model=200 \
    model.norm=batch model.prenorm=True model.layer.dt_min=0.01 model.layer.dt_max=1. \
    +model.layer.temporal="prf_train" +model.layer.spatial="identity"

# psMNIST
python -m train experiment=lra/snn-mnist wandb=null dataset.permute=true loader.num_workers=0 \
    train.monitor=test/accuracy +model.layer.tau=5. model.n_layers=4 model.d_model=200 \
    model.norm=batch model.prenorm=True model.layer.dt_min=0.01 model.layer.dt_max=1. \
    +model.layer.temporal="prf_train" +model.layer.spatial="identity"

# seqCIFAR
python -m train experiment=lra/snn-cifar wandb=null model.layer.fr_scale=0.5 +model.layer.tau=4. \
    model.n_layers=6 model.d_model=128 model.dropout=0. dataset.augment=true dataset.cutout=true \
    model.layer.bidirectional=true +model.layer.temporal='prf_train' +model.layer.spatial='Binary' \
    dataset.grayscale=false optimizer.lr=0.01 optimizer.weight_decay=0.01
```

### Neuron-level ablations (Table 6 and related)

Swap the temporal neuron with `model.layer.temporal` and the spatial neuron with `model.layer.spatial`
on any task, e.g. on ListOps:

```bash
# PRF with an extra reset (Table 9)
python -m train experiment=lra/snn-listops wandb=null loader.num_workers=0 \
    model.layer.fr_scale=12 model.layer.temporal="prf_reset" model.layer.spatial="Binary" \
    train.monitor=test/accuracy

# Baseline neurons: ParaLIF / TC-LIF / TS-LIF / BHRF / ELM / sliding-PSN / LIF
python -m train experiment=lra/snn-listops wandb=null loader.num_workers=0 \
    model.layer.temporal='ParaLIF' train.monitor=test/accuracy
python -m train experiment=lra/snn-listops wandb=null loader.num_workers=0 \
    model.layer.temporal='tclif' train.monitor=test/accuracy
python -m train experiment=lra/snn-listops wandb=null loader.num_workers=0 \
    model.layer.temporal='tslif' train.monitor=test/accuracy
python -m train experiment=lra/snn-listops wandb=null loader.num_workers=0 \
    model.layer.temporal='bhrf' train.monitor=test/accuracy
python -m train experiment=lra/snn-listops wandb=null loader.num_workers=0 \
    model.layer.temporal='elm' train.monitor=test/accuracy
python -m train experiment=lra/snn-listops wandb=null loader.num_workers=0 \
    model.layer.temporal='spsn' model.layer.spatial=null train.monitor=test/accuracy
python -m train experiment=lra/snn-listops wandb=null loader.num_workers=0 \
    +model.layer.tau=2. model.layer.fr_scale=10 model.layer.temporal='lif' model.layer.spatial='Binary' \
    train.monitor=test/accuracy

# Identity (no spatial neuron) variant
python -m train experiment=lra/snn-listops wandb=null loader.num_workers=0 \
    model.layer.fr_scale=12 model.layer.spatial='identity' train.monitor=test/accuracy
```

### Resonance-frequency sensitivity (fr_scale, Supplementary)

```bash
python -m train experiment=lra/snn-cifar wandb=null loader.num_workers=0 model.layer.fr_scale=0.3 \
    train.monitor=test/accuracy dataset.augment=true dataset.cutout=true \
    model.layer.bidirectional=true +model.layer.spatial='Binary'          # fr_scale in {0.15, 0.2, 0.3, 0.35}
python -m train experiment=lra/snn-listops wandb=null loader.num_workers=0 model.layer.fr_scale=13 \
    model.layer.spatial='Binary' train.monitor=test/accuracy             # fr_scale in {11, 13}
python -m train experiment=lra/snn-imdb wandb=null loader.num_workers=0 model.layer.fr_scale=15 \
    model.layer.spatial='Binary' train.monitor=test/accuracy             # fr_scale in {14, ..., 18}
python -m train experiment=lra/snn-aan wandb=null loader.num_workers=0 model.layer.fr_scale=15 \
    model.layer.spatial='Binary' train.monitor=test/accuracy             # fr_scale in {13, 15}
python -m train experiment=lra/snn-pathfinder wandb=null loader.num_workers=0 model.layer.fr_scale=0.7 \
    model.layer.bidirectional=true model.layer.spatial='Binary' train.monitor=test/accuracy \
    encoder=embedding                                                    # fr_scale in {0.5, 0.7}
```

### Firing-rate logging and energy estimation (Table 5)

After training, run inference with firing-rate logging enabled on a saved checkpoint:

```bash
python -m inference experiment=lra/snn-listops wandb=null loader.num_workers=0 \
    model.layer.fr_scale=12 model.layer.spatial='Binary' train.monitor=test/accuracy \
    +model.layer.save_firerate=True train.pretrained_model_path=<path/to/checkpoint.ckpt>
```

`python neuron_dynamic_energy.py` prints the neuron-level dynamic energy estimates
(AC/MAC operations at 45 nm) used for the energy comparison in the paper.

## Repository layout

```
PRF_release/
├── train.py                 # main training entry (Hydra)
├── inference.py             # evaluation / firing-rate analysis entry
├── neuron_dynamic_energy.py # neuron-level dynamic energy estimation
├── configs/                 # Hydra configs (experiments, model, pipeline, dataset, ...)
│   ├── config.yaml
│   ├── experiment/lra/      # snn-listops / snn-imdb / snn-aan / snn-cifar / snn-pathfinder / snn-pathx / snn-mnist
│   ├── model/layer/         # spikingblockgate.yaml (PRF + SD-TCM layer), spikingblock.yaml, ...
│   └── ...
├── src/
│   ├── models/sequence/modules/spikingblock.py   # PRF neuron + SD-TCM blocks
│   ├── models/sequence/modules/{neuron,para_base,bhrf_neuron,elm_neuron}.py  # baseline neurons
│   ├── models/sequence/...  # S4/SaShiMi/Transformer backbones and kernels
│   ├── dataloaders/         # LRA / MNIST / CIFAR / LM datasets
│   └── tasks/, utils/, callbacks/
├── data/getdata.sh          # optional LM corpora download (not needed for the paper experiments)
├── environment.yml          # conda env `prf` (versions used for the paper's experiments)
└── requirements.txt         # pip equivalent of environment.yml
```

## Notes

- The layer used in the paper is `spikingblockgate` (see `configs/model/snn.yaml`); the neuron type is selected with
  `model.layer.temporal` (`null`/`prf` = PRF, `prf_train`, `ParaLIF`, `tclif`, `tslif`, `bhrf`, `elm`, ...) and the
  spatial neuron with `model.layer.spatial` (`Binary`, `identity`, ...).
- This is research code; some optional baseline modules use absolute `from models...` imports. Running from the
  repository root (or setting `PYTHONPATH=.`) avoids import issues.
- Training the LRA tasks requires a GPU; PathX in particular benefits from multiple GPUs.

 
## License

This project is released under the Apache License 2.0 (see `LICENSE`).
