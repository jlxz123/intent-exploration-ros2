# RL Explore Launchers

This directory contains project-local double-click launchers for Linux desktops that open shell scripts as text.

Build the launchers after cloning or moving the project:

```bash
bash launchers/build_launchers.sh
```

Then double-click one of these files:

- `RL_Explore_Auto`: starts Gazebo, Cartographer, RViz, and the autonomous policy.
- `RL_Explore_With_Hand_Intent`: starts Gazebo, Cartographer, RViz, the policy, and the hand-camera intent publisher.

Both launchers resolve the project path from their own location, so the project does not need to live under a fixed `/home/...` path.
