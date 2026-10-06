# Figure 1 reproduction

This folder contains only the code needed to reproduce Figure 1 of Nakkiran et al.,
*Deep Double Descent: Where Bigger Models and More Data Hurt* (JSTAT, 2021).

## Folder layout

```text
Here/
├── data/               # CIFAR-10 is downloaded here when training starts
├── model/
│   └── resnet.py       # width-scaled pre-activation ResNet-18
├── scripts/
│   ├── train.py        # training and checkpointing
│   └── plot.py         # Figure 1 and one summary CSV
├── results/            # generated checkpoints, CSV and figure
└── README.md           # experiment definition and commands
```

No JSON configuration files are used. All experimental settings are visible below and
are also command-line arguments in `scripts/train.py`.

## Experiment represented by Figure 1

| Item | Value |
|---|---|
| Dataset | CIFAR-10, 50,000 train / 10,000 test images |
| Training-label noise | 15%; a selected label is replaced uniformly by one of the other 9 labels |
| Test curve | Expected error on the same 15%-noisy label distribution |
| Model | Pre-activation ResNet-18 |
| Width parameter | every integer `k = 1, 2, ..., 64` |
| Block channels | `[k, 2k, 4k, 8k]`; `k=64` is standard ResNet-18 |
| Repeats | 5 trials per width |
| Optimizer | Adam, learning rate `1e-4`, betas `(0.9, 0.999)`, eps `1e-8` |
| Training | 4,000 epochs, batch size 128, cross-entropy |
| Augmentation | random crop 32 with padding 4, then random horizontal flip |
| Explicit regularization | no weight decay and no dropout |
| Total jobs | 64 widths x 5 trials = 320 |

The full experiment performs 500,480,000 optimizer updates. Run it on CUDA hardware.
The script deliberately refuses to silently replace CUDA with Apple MPS.

## Reproduction choices that the paper does not fully specify

The paper and released author code do not provide every historical random seed and
data-loader detail. This implementation therefore makes the following choices explicit:

- trial `t` uses model seed `t`, noise seed `2000+t`, loader seed `3000+t`, and
  augmentation seed `4000+t`;
- every width in the same trial sees exactly the same noisy labels;
- normalization uses channel mean and standard deviation computed from the complete
  official CIFAR-10 training set;
- evaluation is performed after every epoch so the right panel can be reconstructed;
- PyTorch deterministic algorithms are enabled.

Figure 1 reports error on a noisy test-label distribution, although the stored CIFAR-10
test labels remain clean. For 10 classes and noise probability `p=0.15`, the plotted value is

```text
noisy_test_error = p + (1 - 10*p/9) * clean_test_error
```

## Environment

Use Python 3.10 or newer with PyTorch, torchvision, NumPy and Matplotlib:

```bash
python -m pip install torch torchvision numpy matplotlib
```

## Training commands

Run all 320 jobs sequentially:

```bash
cd /path/to/DD2
python scripts/train.py --widths 1-64 --trials 0-4 --device cuda
```

Split the jobs across several GPUs or machines with `--shard` and `--num-shards`.
Every machine must use the same shared `results/` directory, or its result files must be
copied together afterward:

```bash
python scripts/train.py --widths 1-64 --trials 0-4 --device cuda --shard 0 --num-shards 4
python scripts/train.py --widths 1-64 --trials 0-4 --device cuda --shard 1 --num-shards 4
python scripts/train.py --widths 1-64 --trials 0-4 --device cuda --shard 2 --num-shards 4
python scripts/train.py --widths 1-64 --trials 0-4 --device cuda --shard 3 --num-shards 4
```

Each `(width, trial)` job writes one file such as `results/width_01_trial_0.pt`.
That file contains the model, optimizer and complete epoch history, so rerunning the same
command resumes an interrupted job automatically.

For a short pipeline check on a GPU, explicitly reduce the scope. Such a run is only a
software check and is not Figure 1 evidence:

```bash
python scripts/train.py --widths 1,8,64 --trials 0 --epochs 2 --device cuda
```

## Plotting

After all requested jobs finish:

```bash
python scripts/plot.py
```

This produces only:

- `results/figure1.png`: the final-error panel and the epoch-wise panel;
- `results/summary.csv`: one tidy table containing all recorded epoch metrics.

The plot script warns if any of the expected 320 jobs are missing or incomplete.

