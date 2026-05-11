# rl_explore

[English](README.md) | 中文

`rl_explore` 是一个用于在 ROS 2 / Gazebo 中运行 CCRL 风格探索策略的独立工作空间。项目将训练阶段的网格探索策略迁移到仿真机器人环境中，并支持自主探索和人意图参与探索两种运行方式。

本项目是 [`intent-exploration`](https://github.com/jlxz123/intent-exploration) 的 ROS 2 / Gazebo 仿真部署版本。项目复用其中最终确定的人意图探索策略，并在其训练侧网格环境基础上扩展出 Gazebo、Cartographer、Nav2、RViz、CCRL 风格激光状态构造和人意图输入等 ROS 侧功能。

上游 `intent-exploration` 项目基于 CCRL 探索工作。上游来源、论文引用和许可证说明见 [第三方声明](THIRD_PARTY_NOTICES.md)、[引用](#引用) 和 [许可证](#许可证)。

## 项目内容

从本目录构建 ROS 2 工作空间：

```bash
rosdep install --from-paths src --ignore-src -r -y --skip-keys python3-torch
python -m colcon build --symlink-install
source install/setup.bash
```

工作空间中包含三个自定义 ROS 包：

- `rl_explore_description`：mbot 机器人模型和激光雷达相关 URDF / Xacro 资源。
- `rl_explore_gazebo`：`.npy` 地图、Gazebo 世界生成、Gazebo-ROS bridge 配置、Cartographer 配置、Nav2 配置和 RViz 配置。
- `rl_explore_policy`：最终探索策略模型、checkpoint 加载、ROS 策略节点、键盘意图节点和手势摄像头意图节点。

该工作空间不需要 source `dev_ws`，也不需要导入原始训练项目。

## 主要改动

相比 `intent-exploration`，本项目增加的是 ROS 侧仿真和部署层：

- ROS 2 package 化的 Gazebo 仿真环境。
- 基于 `.npy` 地图生成 Gazebo 世界。
- 与训练环境一致的 CCRL cell 尺度和离散地图表示。
- 面向神经网络输入的 31 束 CCRL 风格激光状态构造。
- 用于 SLAM / RViz 可视化的高分辨率激光链路。
- 使用 Nav2 执行网络输出的离散 waypoint 动作。
- 键盘方向和摄像头食指方向两种人意图发布方式。
- 项目内双击启动器，用于一键启动自主探索或人意图参与探索。

## 地图与几何约定

- `.npy` 地图中的一个 cell 在 Gazebo 中对应 `0.45 m x 0.45 m`。
- `.npy` 中 `0` 表示自由空间。
- `.npy` 中 `255` 表示障碍物。
- `.npy` 中 `128` 表示不可通行区域或边界，在 Gazebo 中同样生成障碍物。
- 障碍物尺寸为 `0.45 x 0.45 x 0.50 m`，中心高度为 `z = 0.25 m`。
- mbot 激光雷达高度约为 `z = 0.105 m`，因此生成的障碍物可以被 `/scan` 扫描到。

需要注意：CCRL cell 是策略输入中的逻辑网格，不等同于 SLAM `/map` 的小栅格分辨率。本项目中神经网络输入由 CCRL 风格激光射线更新得到，而不是直接把 `/map` 强行 reshape 成网络输入。

## 激光输入

仿真机器人中保留两类激光使用方式：

- 高分辨率激光用于 Cartographer、Nav2 和 RViz 可视化。
- CCRL 风格 31 束激光用于更新策略输入中的 cell 地图和 sensor 向量。

CCRL 风格激光范围为机器人正前方左右各 `135 deg`：

```text
-135 deg 到 +135 deg，共 31 束，角度间隔 9 deg
```

策略节点会把激光距离归一化到训练时使用的最大可观测距离。当前最终设置为：

```text
lidar_max_range = 50 cells
cell_size = 0.65 m
最大归一化距离 = 0.65 * 50 = 32.5 m
```

因此 sensor 输入中的激光部分为：

```text
扫描距离 / 32.5
```

## 策略输入与输出

策略网络输入包含两部分：

- `s_map`：四通道 `24 x 24` 图像输入。
- `s_sensor`：32 维向量输入。

`s_map` 的四个通道为：

- `channel 0`：global built map。
- `channel 1`：local built map。
- `channel 2`：global intent map。
- `channel 3`：local frontier intent map。

在没有人意图输入时，`channel 2` 和 `channel 3` 保持为 0。

`s_sensor` 包含：

- 31 束 CCRL 风格激光距离。
- 1 个机器人朝向编码。

网络输出为 3 个离散动作：

- `0`：左转到下一个 90 度方向。
- `1`：沿当前朝向前进一个逻辑 cell。
- `2`：右转到下一个 90 度方向。

## 动作执行

网络输出不会直接转换成固定时间的速度命令，而是转换成下一个导航目标。策略节点根据当前位姿构造离散 waypoint，并通过 Nav2 执行导航。

这种方式比简单发送固定 `/cmd_vel` 更接近训练时的横平竖直网格动作，也减少了累计姿态误差对策略执行的影响。

## 运行方式

项目提供两个可双击启动的 Linux 可执行文件：

```text
launchers/RL_Explore_Auto
launchers/RL_Explore_With_Hand_Intent
```

双击 `RL_Explore_Auto` 会启动自主探索流程：

- `RL Explore Gazebo`：启动 Gazebo、Cartographer、Nav2 和 RViz。
- `RL Explore Policy`：等待环境加载后启动策略节点。

双击 `RL_Explore_With_Hand_Intent` 会启动人意图参与探索流程：

- `RL Explore Gazebo`：启动 Gazebo、Cartographer、Nav2 和 RViz。
- `RL Explore Policy`：启动策略节点。
- `RL Explore Hand Intent`：启动摄像头手势意图发布节点。

启动器是 ELF 可执行文件，而不是 `.desktop` 或 `.sh` 文件。这样可以避免某些 Linux 桌面环境双击脚本时只把它当作文本打开。

如果移动项目目录或重新 clone 项目，可以重新构建启动器：

```bash
bash launchers/build_launchers.sh
```

也可以从终端直接运行并指定地图或设备：

```bash
scripts/run_rl_explore_all.sh train_map_l2 cuda
scripts/run_rl_explore_all.sh test_map_l1 cpu
```

手势意图参与版本：

```bash
scripts/run_rl_explore_with_hand_intent.sh train_map_l2 cuda
```

## Python 与 CUDA 环境

ROS 包构建不会自动安装所有 Python 运行依赖。基础 ROS 依赖可用：

```bash
rosdep install --from-paths src --ignore-src -r -y --skip-keys python3-torch
```

运行策略节点所需的 Python 依赖包括：

- `torch`
- `numpy`
- `opencv-python` 或系统 OpenCV Python 绑定
- 手势节点需要的 MediaPipe 相关依赖

本机当前使用的 CUDA PyTorch venv 位于：

```text
/home/zbf/roslearn/ros_torch_cuda_venv
```

该 venv 使用 `--system-site-packages` 创建，因此可以使用系统 ROS Python 包，同时把 CUDA PyTorch 保持在独立虚拟环境中。构建本工作空间时建议先激活该 venv：

```bash
source /opt/ros/humble/setup.bash
source /home/zbf/roslearn/ros_torch_cuda_venv/bin/activate
python -m colcon build --symlink-install
```

## 常用 ROS 命令

单独启动 Gazebo / Cartographer / Nav2 / RViz：

```bash
source install/setup.bash
ros2 launch rl_explore_gazebo rl_explore_gazebo.launch.py map_name:=train_map_l3
```

单独启动策略节点：

```bash
source install/setup.bash
ros2 launch rl_explore_policy rl_explore_policy.launch.py device:=cuda
```

发布一次意图方向：

```bash
ros2 topic pub --once /rl_explore/intent_direction std_msgs/msg/Int32 "{data: 0}"
```

清除意图：

```bash
ros2 topic pub --once /rl_explore/clear_intent std_msgs/msg/Empty "{}"
```

## 引用

本项目基于 `intent-exploration` 进行 ROS 2 / Gazebo 侧扩展：

```bibtex
@misc{intent_exploration,
  title = {intent-exploration},
  author = {{jlxz123}},
  year = {2026},
  url = {https://github.com/jlxz123/intent-exploration}
}
```

上游 `intent-exploration` 项目基于 CCRL 探索论文：

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

上游训练项目仓库：

```text
https://github.com/jlxz123/intent-exploration
```

## 许可证

本项目使用 MIT License 发布。部分代码、模型结构思想、状态表示和训练/评估概念来源于或改编自 `jlxz123/intent-exploration`，其上游 MIT 许可证声明保存在 [LICENSES/Intent_Exploration-MIT.txt](LICENSES/Intent_Exploration-MIT.txt)。
