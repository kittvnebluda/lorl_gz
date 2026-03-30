# Legged RL in Gazebo Harmonic (WIP)

Reinforcement learning workspace for quadruped locomotion and obstacle traversal,
focused on Unitree Go2 workflows with ROS 2 and Gazebo Harmonic.

## Project Scope

- Train and evaluate policies with Stable-Baselines3.
- Run policies through the `scripts` CLI (`train`, `run`, `reset`).
- Store checkpoints, final models, and TensorBoard logs in this repository.

## Repository Layout

- `scripts/` - CLI entrypoints for training, testing, random policy, and reset tools.
- `src/` - ROS 2 packages and the Gymnasium environment implementation.

## Quickstart

TODO

## Install and Build

### Prerequisites

TODO

### Workspace Build

From repository root:

```bash
colcon build
source install/setup.bash
```

## CLI Usage

The command interface is implemented in `scripts/__main__.py`.

### Train

```bash
python -m scripts train [--name NAME] [--algo {ppo,sac}] [--total_timesteps N] \
  [--load_model MODEL.zip] [--step_delay S] [--reset_delay S] [--real_time_factor RTF]
```

- `--name` sets model filename prefix.
- `--algo` selects algorithm (`ppo` or `sac`).
- `--load_model` resumes from a saved archive.
- `--step_delay`, `--reset_delay`, `--real_time_factor` tune runtime pacing.

### Run trained model

```bash
python -m scripts run model <MODEL_PATH> [--algo ALGO] [--debug]
```

### Run random policy

```bash
python -m scripts run random [--debug]
```

### Reset agent

```bash
python -m scripts reset
```

## Training Notes

- Training creates:
  - `./models/<name>_<algo>.zip` (final model)
  - `./checkpoints/` (periodic checkpoints)
  - `./tb_logs/gazebo_<algo>/` (TensorBoard logs)
- Current training code hard-requires Intel XPU:
  - `assert torch.xpu.is_available()`
  - `params["device"] = "xpu"`

If your machine does not provide XPU support, adjust `scripts/train.py` device handling before training.

## Acknowledgements

- Two maps originate from the [ICRA 2024 repository](https://github.com/teamgrit-lab/ICRA2024_Quadruped_Robot_Challenges.git).
- For Unitree Go2, this project uses a fork of [unitree_go2_ros2](https://github.com/kittvnebluda/unitree_go2_ros2.git).
