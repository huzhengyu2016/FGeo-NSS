# FGeo-NSS

This is the official implementation of the paper "A Hybrid Framework for Automated Geometric Problem-Solving by
Integrating Formal Symbolic Systems and Deep Learning". We construct a neuro-symbolic system to solve geometric problems
efficiently. The neural component employs a gating-enhanced attention network to provide directional guidance for the
solving process, enabling effective heuristics and pruning of theorem branches. The symbolic component is a
bidirectional solver built upon FormalGeo, which performs rigorous geometric relational reasoning and algebraic
computation. The neural component predicts the theorems required to solve the problem in its current state, while the
symbolic component applies these theorems and updates the problem state; the two parts interact iteratively until the
problem is solved. In this system, the solving process is organized as a graph structure where Facts serve as nodes and
theorems as edges, thereby generating a human-readable solution.

## Running

Create a new Python environment and install dependencies:

    $ conda create -n NSS python=3.12.12
    $ conda activate NSS
    $ cd FGeo-NSS
    $ pip install -e .

Download the dataset and checkpoints from
[Google Drive](https://drive.google.com/file/d/1Rziz2vaXKUsVaaRTmaJm9SUFPOK50oB5/view?usp=drive_link), and extract them
to the current project. Now your directory structure should look like:

```
|--datasets
|  |--diagram
|  |--ggbs
|  |--problems
|  |--gdl.json
|
|--outputs
|  |--checkpoints
|  |--files
|  |--log
|  |--synthetic_data
|
|--src
|  |--nss
|     |--data.py
|     |--dataset.py
|     |--model.py
|     |--problem.py
|     |--solve.py
|     |--tests.py
|     |--tools.py
|     |--train.py
|
|--.gitignore
|
|--config.json
|
|--LICENSE
|
|--pyproject.toml
|
|--README.md
```

Obtain experimental results:

    $ conda activate NSS
    $ cd FGeo-NSS
    $ python tools.py --func get_acc

You will see output similar to the following:

| Method           | Total | L1    | L2    | L3    | L4    | L5    | L6    |
|------------------|-------|-------|-------|-------|-------|-------|-------|
| T5-small         | 36.14 | 47.66 | 40.0  | 35.45 | 22.54 | 15.52 | 1.96  |
| BART-base        | 54.0  | 73.83 | 56.39 | 48.88 | 35.21 | 25.86 | 9.8   |
| Inter-GPS        | 60.5  | 77.66 | 64.44 | 57.09 | 44.37 | 34.48 | 13.73 |
| NGS              | 62.6  | 65.07 | 68.15 | 66.45 | 58.11 | 54.29 | 37.5  |
| DualGeoSolver    | 62.11 | 65.75 | 67.52 | 67.1  | 59.46 | 42.86 | 35.42 |
| FGeo-DRL         | 80.85 | 97.97 | 88.36 | 72.61 | 60.39 | 55.42 | 39.62 |
| FGeo-HyperGNet   | 88.36 | 95.96 | 94.17 | 90.67 | 74.65 | 65.52 | 58.82 |
| Forward-BFS      | 38.86 | 56.87 | 39.21 | 32.81 | 22.77 | 11.69 | 6.32  |
| Forward-DFS      | 36.16 | 52.51 | 40.42 | 24.72 | 18.1  | 17.53 | 8.8   |
| Forward-RS       | 39.71 | 55.52 | 42.1  | 35.41 | 20.88 | 12.34 | 8.8   |
| Backward-BFS     | 35.44 | 67.95 | 35.43 | 12.4  | 8.32  | 3.57  | 1.35  |
| Backward-DFS     | 33.73 | 65.72 | 32.76 | 11.73 | 7.45  | 3.25  | 1.13  |
| Backward-RS      | 34.05 | 66.94 | 32.55 | 11.58 | 7.3   | 3.57  | 1.13  |
| DeepSeek v3      | 60.79 | 73.19 | 58.89 | 61.19 | 45.07 | 43.1  | 41.18 |
| Kimi-K2          | 82.6  | 80.77 | 83.6  | 83.48 | 83.47 | 86.27 | 81.25 |
| Ours             | 89.63 | 96.92 | 93.06 | 92.41 | 80.99 | 72.55 | 48.44 |
| Text-only-1      | 62.3  | 83.33 | 66.88 | 56.7  | 35.54 | 17.65 | 17.19 |
| Text-only-3      | 69.41 | 86.41 | 71.92 | 70.98 | 42.98 | 37.25 | 23.44 |
| Text-only-5      | 70.78 | 87.95 | 73.82 | 70.09 | 50.41 | 37.25 | 18.75 |
| Forward-only-1   | 62.64 | 80.77 | 64.35 | 58.04 | 42.98 | 31.37 | 21.88 |
| Forward-only-3   | 70.69 | 84.62 | 77.92 | 69.2  | 50.41 | 31.37 | 25.0  |
| Forward-only-5   | 71.38 | 84.87 | 79.5  | 69.64 | 52.89 | 31.37 | 21.88 |
| No Gate-1        | 65.38 | 82.56 | 67.51 | 63.39 | 47.93 | 33.33 | 15.62 |
| No Gate-3        | 71.72 | 84.36 | 77.6  | 71.88 | 56.2  | 39.22 | 20.31 |
| No Gate-5        | 70.18 | 84.36 | 76.97 | 67.86 | 53.72 | 31.37 | 20.31 |
| Small Model-1    | 67.52 | 83.85 | 70.03 | 65.18 | 47.93 | 39.22 | 23.44 |
| Small Model-3    | 73.95 | 89.74 | 78.55 | 72.32 | 56.2  | 37.25 | 23.44 |
| Small Model-5    | 74.81 | 90.0  | 77.29 | 76.34 | 57.85 | 37.25 | 26.56 |
| Standard Model-1 | 72.07 | 85.9  | 73.19 | 74.11 | 60.33 | 41.18 | 21.88 |
| Standard Model-3 | 77.38 | 91.79 | 79.81 | 79.91 | 66.94 | 39.22 | 18.75 |
| Standard Model-5 | 78.15 | 94.1  | 81.39 | 79.46 | 63.64 | 39.22 | 18.75 |

If you want to synthesize data and train the model from scratch, run the following commands. First, synthesize data:

    $ conda activate NSS
    $ cd FGeo-NSS/src/nss
    $ python data.py --func generate_synthetic_data
    $ python data.py --func make_training_data

Train model:

    $ conda activate NSS
    $ cd FGeo-NSS/src/nss
    $ python train.py
    $ python train.py --text_only
    $ python train.py --forward_only
    $ python train.py --no_gate
    $ python train.py --small_model

Run PAC for testing. Results will be saved to `outputs/log/`:

    $ conda activate NSS
    $ cd FGeo-NSS/src/nss
    $ python solve.py --timeout 600
    $ python solve.py
    $ python solve.py --text_only
    $ python solve.py --forward_only
    $ python solve.py --no_gate
    $ python solve.py --small_model

## Citation

coming soon...




