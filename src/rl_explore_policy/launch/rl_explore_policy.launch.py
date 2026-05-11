from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description():
    default_checkpoint = str(Path(get_package_share_directory("rl_explore_policy")) / "checkpoints" / "final_dagger.pth")
    return LaunchDescription(
        [
            DeclareLaunchArgument("checkpoint_path", default_value=default_checkpoint),
            DeclareLaunchArgument("device", default_value="cuda"),
            DeclareLaunchArgument("world_name", default_value="train_map_l1"),
            DeclareLaunchArgument("spawn_base_seed", default_value="1"),
            DeclareLaunchArgument("spawn_eval_index", default_value="0"),
            DeclareLaunchArgument("debug_dump_dir", default_value=""),
            Node(
                package="rl_explore_policy",
                executable="policy_node",
                name="rl_explore_policy_node",
                output="screen",
                parameters=[
                    {
                        "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                        "device": LaunchConfiguration("device"),
                        "world_name": LaunchConfiguration("world_name"),
                        "model_name": "mbot",
                        "action_execution_mode": "nav2",
                        "cmd_vel_topic": "/rl_explore/manual_cmd_vel",
                        "nav_action_name": "navigate_to_pose",
                        "static_map_package": "rl_explore_gazebo",
                        "static_map_name": LaunchConfiguration("world_name"),
                        "world_spawn_x": "auto",
                        "world_spawn_y": "auto",
                        "world_spawn_yaw": "auto",
                        "world_spawn_base_seed": LaunchConfiguration("spawn_base_seed"),
                        "world_spawn_eval_index": LaunchConfiguration("spawn_eval_index"),
                        "ccrl_map_rows": 80,
                        "ccrl_map_cols": 80,
                        "cell_size_m": 0.65,
                        "plain_laser_range_max_cells": 50.0,
                        "action_timeout": 20.0,
                        "post_action_pause": 0.03,
                        "debug_dump_dir": LaunchConfiguration("debug_dump_dir"),
                        "show_ccrl_map_window": True,
                        "ccrl_map_window_refresh_hz": 2.0,
                        "ccrl_map_window_scale": 10,
                    }
                ],
            ),
        ]
    )
