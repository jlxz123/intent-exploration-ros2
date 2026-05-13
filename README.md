# rl_explore

English | [中文](README_zh.md)

ROS 2 / Gazebo workspace for running a CCRL-style DAgger exploration policy with optional human intent input.

This project is a ROS 2 / Gazebo adaptation of the [`intent-exploration`](https://github.com/jlxz123/intent-exploration) project. It reuses the final intent-conditioned exploration policy and extends the training-side grid environment into a deployable robot simulation pipeline with Gazebo, Cartographer, Nav2, RViz, CCRL-style lidar state construction, and optional human intent input.

The upstream `intent-exploration` project is based on the CCRL exploration work. See [Third-Party Notices](THIRD_PARTY_NOTICES.md), [Citation](#citation), and [License](#license).

## Workspace

Build from this directory:

```bash
rosdep install --from-paths src --ignore-src -r -y --skip-keys python3-torch
python -m colcon build --symlink-install
source install/setup.bash
```

The workspace contains the custom resources needed for this ROS/Gazebo experiment:

- `rl_explore_description`: mbot lidar model resources with CCRL-compatible lidar FOV.
- `rl_explore_gazebo`: `.npy` maps, generated Gazebo worlds, bridge config, Cartographer config, Nav2 config, and RViz config.
- `rl_explore_policy`: final policy model, trained checkpoint, ROS policy node, keyboard intent node, and hand-camera intent node.

It should not require sourcing `dev_ws` or importing the original training project.

## Project Contributions

Compared with `intent-exploration`, this repository adds the ROS-side simulation and deployment layer:

- ROS 2 package structure for Gazebo-based simulation.
- `.npy` map to Gazebo world generation with CCRL cell scaling.
- CCRL-style 31-beam lidar state construction for neural-network input.
- Separate high-resolution lidar usage for SLAM / visualization.
- Discrete waypoint execution of learned actions through ROS navigation logic.
- Keyboard and hand-camera intent direction publishers.
- A two-panel intent exploration GUI that combines the hand-camera view and the CCRL mapping view.
- Project-local Linux launchers for one-click autonomous or human-intent runs.

## Important Geometry

- One `.npy` map cell is `0.45 m x 0.45 m` in Gazebo.
- `0` cells are free space.
- `255` cells are obstacles.
- `128` cells are invalid/boundary and are also generated as obstacles.
- Obstacle boxes are `0.45 x 0.45 x 0.50 m`, centered at `z=0.25 m`.
- The mbot lidar is at about `z=0.105 m`, so generated obstacles are visible to `/scan`.

## Lidar

The Gazebo lidar itself is restricted to the CCRL front-facing field of view:

- `angle_min = -2.35619 rad`
- `angle_max = 2.35619 rad`
- `samples = 271`

The policy node then downsamples this `/scan` to the 31 CCRL beams:

```text
-135 deg, step 9 deg, 31 beams total
```

## Policy Motion

Network outputs are executed as discrete waypoint actions:

- `0`: snap to the nearest cardinal yaw, then turn left to the next 90 degree yaw.
- `1`: move to the next `0.45 m` logical cell along the current cardinal yaw.
- `2`: snap to the nearest cardinal yaw, then turn right to the next 90 degree yaw.

The controller tracks these target yaw / target cell waypoints using `/odom`, instead of just sending a fixed velocity for a fixed time. This keeps the simulated robot much closer to the grid-aligned motion used during CCRL training.

## Run

Project-local double-click launchers:

```text
launchers/RL_Explore_Auto
launchers/RL_Explore_With_Hand_Intent
```

Double-click `RL_Explore_Auto` to open two terminals:

- `RL Explore Gazebo`: starts Gazebo + Cartographer with the selected map.
- `RL Explore Policy`: waits 12 seconds, then starts the policy node with `device:=cuda`.

Double-click `RL_Explore_With_Hand_Intent` to open the same two terminals plus:

- `RL Explore Hand Intent`: starts the camera-based hand intent publisher.
- `RL Explore Intent GUI`: opens a two-panel GUI with the camera capture view and the CCRL mapping view.

In hand-intent mode, the standalone hand-camera and CCRL map debug windows are disabled and replaced by the combined GUI. Gazebo, RViz, Cartographer, Nav2, and terminal windows still run normally.

The launchers run:

```bash
scripts/run_rl_explore_all.sh
scripts/run_rl_explore_with_hand_intent.sh
```

They are ELF binaries instead of `.desktop` or `.sh` files, because some Linux desktops open scripts as text when double-clicked. If the project is moved or cloned to another path, rebuild them with:

```bash
bash launchers/build_launchers.sh
```

To choose another map or device from a terminal:

```bash
scripts/run_rl_explore_all.sh test_map_l1 cpu
```

Use the CUDA torch venv before building or running the policy node:

```bash
source /opt/ros/humble/setup.bash
source /home/zbf/roslearn/ros_torch_cuda_venv/bin/activate
```

Start Gazebo and Cartographer:

```bash
ros2 launch rl_explore_gazebo rl_explore_gazebo.launch.py map_name:=train_map_l3
```

Start the policy node in another terminal:

```bash
source install/setup.bash
ros2 launch rl_explore_policy rl_explore_policy.launch.py device:=cuda
```

Use `device:=cpu` only when CUDA is unavailable or when debugging without GPU access.

Set an intent direction:

```bash
ros2 topic pub --once /rl_explore/intent_direction std_msgs/msg/Int32 "{data: 0}"
```

Clear intent:

```bash
ros2 topic pub --once /rl_explore/clear_intent std_msgs/msg/Empty "{}"
```

## Python Dependencies

The ROS package build does not install missing system dependencies by itself. Install them with:

```bash
rosdep install --from-paths src --ignore-src -r -y --skip-keys python3-torch
```

The runtime Python must provide:

- `torch`
- `numpy`
- `opencv-python` or system OpenCV Python bindings

On this machine, the CUDA PyTorch runtime is installed in:

```text
/home/zbf/roslearn/ros_torch_cuda_venv
```

This venv was created with `--system-site-packages`, so it can use ROS Python packages from the system installation while keeping CUDA PyTorch local to the venv. Build with `python -m colcon build --symlink-install` while the venv is active, so installed ROS Python entry points use the venv interpreter.

## Citation

This project builds on `intent-exploration`:

```bibtex
@misc{intent_exploration,
  title = {intent-exploration},
  author = {{jlxz123}},
  year = {2026},
  url = {https://github.com/jlxz123/intent-exploration}
}
```

The upstream intent-exploration project is based on the CCRL exploration paper:

```bibtex
@article{li2023autonomous,
  title = {Autonomous Exploration and Mapping for Mobile Robots via Cumulative Curriculum Reinforcement Learning},
  author = {Li, Zhi and Xin, Jinghao and Li, Ning},
  journal = {arXiv preprint arXiv:2302.13025},
  doi = {10.48550/arXiv.2302.13025},
  url = {https://arxiv.org/abs/2302.13025},
  year = {2023}
}
```

Upstream training project:

```text
https://github.com/jlxz123/intent-exploration
```

## License

This repository is released under the MIT License. Portions derived from or adapted from `jlxz123/intent-exploration` remain subject to the upstream MIT license notice preserved in [LICENSES/Intent_Exploration-MIT.txt](LICENSES/Intent_Exploration-MIT.txt).
