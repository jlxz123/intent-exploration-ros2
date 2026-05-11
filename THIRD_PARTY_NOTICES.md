# Third-Party Notices

## intent-exploration

This project includes code, model-architecture ideas, state-representation
logic, checkpoint-loading logic, map conventions, trained-policy conventions,
and interaction concepts derived from or adapted from the intent-exploration
project:

- Repository: https://github.com/jlxz123/intent-exploration
- License: MIT License, copyright (c) 2023 BeamanLi and copyright (c) 2026 jlxz123.

The intent-exploration project is itself based on the CCRL exploration work:

- Paper: Zhi Li, Jinghao Xin, Ning Li, "Autonomous Exploration and Mapping for
  Mobile Robots via Cumulative Curriculum Reinforcement Learning",
  arXiv:2302.13025, 2023.

The upstream MIT license text is preserved in:

```text
LICENSES/Intent_Exploration-MIT.txt
```

This repository modifies and extends the intent-exploration policy and state
representation for ROS 2 / Gazebo simulation, CCRL-style lidar emulation,
discrete waypoint execution, human intent interaction, camera-based hand intent
publishing, and project-local launcher packaging.

## ROS 2 And Gazebo Dependencies

The ROS 2, Gazebo, Cartographer, Nav2, and related packages used by this
workspace are external dependencies. Their source code is not redistributed in
this repository and remains governed by their respective upstream licenses.
