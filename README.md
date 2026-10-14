# Learning Architecture as Regulation

Code for *Learning Architecture as Regulation: Comparing Reinforcement Learning Architectures for
Competitive Ride-Hailing Pricing* (Ndayambaje, Feo-Flushing and Indaco).

Several ride-hailing platforms share one city and one pool of riders, and each learns its own prices
with TD3 to maximize profit. We compare four ways of organizing that learning:

| Architecture | Flag | Idea |
|---|---|---|
| Local RL | `local` | Each platform learns on its own |
| Regulator-weighted FRL | `frl` | A regulator scores each platform's actor update and averages them |
| Personalized FRL | `pfrl` | Same averaging, but each platform keeps part of its own actor |
| Hierarchical RL | `hrl` | A slower regulator sends a guidance signal that platforms see as input |

## Previous Work

The simulator extends work previously done by Lei & Ukkusuri (2023).

> Lei, Z. and Ukkusuri, S. V. (2023). Scalable reinforcement learning approaches for dynamic pricing
> in ride-hailing systems. *Transportation Research Part B*, 178, 102848.
> [doi:10.1016/j.trb.2023.102848](https://doi.org/10.1016/j.trb.2023.102848)

The inputs for their NYC scenario (used for our 242-zone experiments) are on
[Google Drive](https://drive.google.com/drive/folders/1IVQUZXHgDQMe1IrVX2A60bidymF0Ccbq?usp=share_link).
Put them in `nyc_small/`. The 4×4 and 8×8 grids are already in `grid_small_dynamic/` on in this repository.

## Requirements

Python 3.10+ (tested on 3.12) with:

```bash
pip install torch numpy pandas scipy matplotlib tensorboard
```

## Code

| File | What it does |
|---|---|
| `main.py` | Entry point and all command-line options |
| `sim.py` | City simulator: vehicles, riders, matching, driver supply |
| `center.py`, `pricer.py`, `model.py` | Platform, TD3 learner and networks |
| `demand.py` | Logit split of riders across platforms |
| `runner.py` | Training and test loops |
| `federation.py` | Regulator-weighted and personalized FRL |
| `regulator.py` | HRL regulator |
| `metrics.py` | Rider-surplus index, price volatility and dispersion, Gini |
| `run_experiments.sh` | Runs every experiment in the paper |

## Running

Each run trains (`-m all`) and then tests on 5 held-out episodes (`-m test`) with the same flags.
The common settings:

```bash
TD3="--pricing_alg TD3_MLP -alr 0.00001 -clr 0.001 --batch_size 32 -sa Gaussian -pd 0 -pe 64 -ac"
SETUP="-d nyc_small/ --competitive --no_outside_option --n_epochs 10 --seed 5 -n s5"
```

**Local RL**
```bash
python main.py $TD3 $SETUP --architecture local -m all
python main.py $TD3 $SETUP --architecture local -m test
```

**Regulator-weighted FRL**
```bash
python main.py $TD3 $SETUP --architecture frl --lambda_rs 1.0 --tau_frl 0.5 -m all
python main.py $TD3 $SETUP --architecture frl --lambda_rs 1.0 --tau_frl 0.5 -m test
```

**Personalized FRL**
```bash
python main.py $TD3 $SETUP --architecture pfrl --share_beta 0.4 -m all
python main.py $TD3 $SETUP --architecture pfrl --share_beta 0.4 -m test
```

**Hierarchical RL**
```bash
python main.py $TD3 $SETUP --architecture hrl --hrl_interval 10 -m all
python main.py $TD3 $SETUP --architecture hrl --hrl_interval 10 -m test
```

Add `--num_platforms 3` to change the number of platforms, or `-c cuda` to use a GPU.
Run `python main.py -h` to see every option.

**Everything at once** (5 seeds, all architectures, ablations, platform and grid scaling):
```bash
bash run_experiments.sh all nyc_small/ grid_small_dynamic/4/4/ grid_small_dynamic/8/8/
```


Results are written to `results/<scenario>/competitive_<architecture>/`.

