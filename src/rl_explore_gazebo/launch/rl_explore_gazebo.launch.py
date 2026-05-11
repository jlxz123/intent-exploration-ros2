import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
import numpy as np
import xacro


CELL_SIZE_M = 0.65
CCRL_YAWS = (0.0, -0.5 * np.pi, np.pi, 0.5 * np.pi)


def _write_bridge_config(world_name):
    config_path = Path("/tmp") / f"rl_explore_ros_gz_bridge_{world_name}.yaml"
    config_path.write_text(
        f"""---
- ros_topic_name: "/cmd_vel"
  gz_topic_name: "/rl_explore_cmd_vel"
  ros_type_name: "geometry_msgs/msg/Twist"
  gz_type_name: "gz.msgs.Twist"
  direction: ROS_TO_GZ
- ros_topic_name: "/clock"
  gz_topic_name: "/clock"
  ros_type_name: "rosgraph_msgs/msg/Clock"
  gz_type_name: "gz.msgs.Clock"
  direction: GZ_TO_ROS
- ros_topic_name: "/odom_raw"
  gz_topic_name: "/model/mbot/odometry"
  ros_type_name: "nav_msgs/msg/Odometry"
  gz_type_name: "gz.msgs.Odometry"
  direction: GZ_TO_ROS
- ros_topic_name: "/joint_states"
  gz_topic_name: "/world/{world_name}/model/mbot/joint_state"
  ros_type_name: "sensor_msgs/msg/JointState"
  gz_type_name: "gz.msgs.Model"
  direction: GZ_TO_ROS
- ros_topic_name: "/scan"
  gz_topic_name: "/lidar"
  ros_type_name: "sensor_msgs/msg/LaserScan"
  gz_type_name: "gz.msgs.LaserScan"
  direction: GZ_TO_ROS
- ros_topic_name: "/ccrl_scan"
  gz_topic_name: "/ccrl_lidar"
  ros_type_name: "sensor_msgs/msg/LaserScan"
  gz_type_name: "gz.msgs.LaserScan"
  direction: GZ_TO_ROS
""",
        encoding="utf-8",
    )
    return str(config_path)


def _map_cell_to_world(row, col, rows, cols):
    x = (float(col) + 0.5 - float(cols) * 0.5) * CELL_SIZE_M
    y = (float(rows) * 0.5 - float(row) - 0.5) * CELL_SIZE_M
    return x, y


def _map_index_for_eval_seed(map_name):
    try:
        level = int(str(map_name).rsplit("_l", 1)[1])
    except (IndexError, ValueError):
        return 0
    return max(0, level - 1)


def _ccrl_spawn_seed(map_name, base_seed, eval_index):
    return int(base_seed) + _map_index_for_eval_seed(map_name) * 1000 + int(eval_index)


def _ccrl_random_spawn_pose(map_path, map_name, base_seed, eval_index):
    grid = np.load(map_path)
    free_cells = np.argwhere(grid == 0)
    if free_cells.size == 0:
        row, col = 0, 0
    else:
        rng = np.random.default_rng(_ccrl_spawn_seed(map_name, base_seed, eval_index))
        orientation = int(rng.integers(4))
        row, col = free_cells[int(rng.integers(len(free_cells)))]
        row, col = int(row), int(col)
        x, y = _map_cell_to_world(row, col, int(grid.shape[0]), int(grid.shape[1]))
        return x, y, float(CCRL_YAWS[orientation]), row, col, orientation, _ccrl_spawn_seed(map_name, base_seed, eval_index)

    orientation = 0
    x, y = _map_cell_to_world(row, col, int(grid.shape[0]), int(grid.shape[1]))
    return x, y, float(CCRL_YAWS[orientation]), int(row), int(col), orientation, _ccrl_spawn_seed(map_name, base_seed, eval_index)


def _launch_setup(context, *args, **kwargs):
    gazebo_pkg = get_package_share_directory("rl_explore_gazebo")
    description_pkg = get_package_share_directory("rl_explore_description")

    map_name = LaunchConfiguration("map_name").perform(context)
    world_path = os.path.join(gazebo_pkg, "worlds", f"{map_name}.sdf")
    map_path = os.path.join(gazebo_pkg, "maps", f"{map_name}.npy")
    if not os.path.exists(world_path):
        raise FileNotFoundError(f"World not found: {world_path}")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Map not found: {map_path}")

    spawn_x_arg = LaunchConfiguration("spawn_x").perform(context)
    spawn_y_arg = LaunchConfiguration("spawn_y").perform(context)
    spawn_yaw_arg = LaunchConfiguration("spawn_yaw").perform(context)
    spawn_base_seed = int(LaunchConfiguration("spawn_base_seed").perform(context))
    spawn_eval_index = int(LaunchConfiguration("spawn_eval_index").perform(context))
    if spawn_x_arg == "auto" or spawn_y_arg == "auto":
        spawn_x, spawn_y, auto_yaw, spawn_row, spawn_col, spawn_orientation, spawn_seed = _ccrl_random_spawn_pose(
            map_path,
            map_name,
            spawn_base_seed,
            spawn_eval_index,
        )
    else:
        spawn_x, spawn_y = float(spawn_x_arg), float(spawn_y_arg)
        auto_yaw = 0.0
        spawn_row, spawn_col, spawn_orientation = -1, -1, -1
        spawn_seed = _ccrl_spawn_seed(map_name, spawn_base_seed, spawn_eval_index)
    if spawn_yaw_arg == "auto":
        spawn_yaw = auto_yaw
    else:
        spawn_yaw = float(spawn_yaw_arg)

    xacro_path = os.path.join(description_pkg, "urdf", "mbot_with_lidar_gazebo_harmonic.xacro")
    robot_description = xacro.process_file(xacro_path).toxml()

    ros_gz_sim_pkg = get_package_share_directory("ros_gz_sim")
    nav2_bringup_pkg = get_package_share_directory("nav2_bringup")
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim_pkg, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": "-r " + world_path}.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic",
            "robot_description",
            "-name",
            "mbot",
            "-x",
            f"{spawn_x:.6f}",
            "-y",
            f"{spawn_y:.6f}",
            "-z",
            LaunchConfiguration("spawn_z"),
            "-Y",
            f"{spawn_yaw:.6f}",
        ],
        output="screen",
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[
            {
                "config_file": _write_bridge_config(map_name),
            }
        ],
        output="screen",
    )

    odom_tf = Node(
        package="rl_explore_gazebo",
        executable="odom_tf_publisher",
        name="rl_explore_odom_tf_publisher",
        output="screen",
        parameters=[
            {
                "odom_topic": "/odom_raw",
                "normalized_odom_topic": "/odom",
                "parent_frame": "odom",
                "child_frame": "base_footprint",
                "use_sim_time": True,
            }
        ],
    )

    cartographer = Node(
        condition=IfCondition(LaunchConfiguration("start_slam")),
        package="cartographer_ros",
        executable="cartographer_node",
        name="cartographer_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=[
            "-configuration_directory",
            os.path.join(gazebo_pkg, "config"),
            "-configuration_basename",
            "lds_2d.lua",
        ],
    )

    occupancy_grid = Node(
        condition=IfCondition(LaunchConfiguration("start_slam")),
        package="cartographer_ros",
        executable="cartographer_occupancy_grid_node",
        name="cartographer_occupancy_grid_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=["-resolution", LaunchConfiguration("slam_resolution"), "-publish_period_sec", "0.2"],
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_pkg, "launch", "navigation_launch.py")),
        condition=IfCondition(LaunchConfiguration("start_nav2")),
        launch_arguments={
            "use_sim_time": "true",
            "autostart": "true",
            "params_file": os.path.join(gazebo_pkg, "config", "rl_explore_nav2_params.yaml"),
            "use_composition": "False",
            "namespace": "",
        }.items(),
    )

    rviz_config_arg = LaunchConfiguration("rviz_config").perform(context)
    if rviz_config_arg == "nav2_default":
        rviz_config = os.path.join(nav2_bringup_pkg, "rviz", "nav2_default_view.rviz")
    else:
        rviz_config = rviz_config_arg

    rviz = Node(
        condition=IfCondition(LaunchConfiguration("start_rviz")),
        package="rviz2",
        executable="rviz2",
        name="rl_explore_rviz",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=["-d", rviz_config],
    )

    return [
        gazebo,
        LogInfo(
            msg=(
                f"[rl_explore] spawn map={map_name}, cell=({spawn_row},{spawn_col}), "
                f"world=({spawn_x:.3f},{spawn_y:.3f}), yaw={spawn_yaw:.3f}, "
                f"orientation={spawn_orientation}, seed={spawn_seed}, "
                f"base_seed={spawn_base_seed}, eval_index={spawn_eval_index}, "
                f"cell_size={CELL_SIZE_M:.2f}m"
            )
        ),
        robot_state_publisher,
        spawn_entity,
        bridge,
        odom_tf,
        cartographer,
        occupancy_grid,
        navigation,
        rviz,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("map_name", default_value="train_map_l1"),
            DeclareLaunchArgument("spawn_x", default_value="auto"),
            DeclareLaunchArgument("spawn_y", default_value="auto"),
            DeclareLaunchArgument("spawn_z", default_value="0.0"),
            DeclareLaunchArgument("spawn_yaw", default_value="auto"),
            DeclareLaunchArgument("spawn_base_seed", default_value="1"),
            DeclareLaunchArgument("spawn_eval_index", default_value="0"),
            DeclareLaunchArgument("start_slam", default_value="true"),
            DeclareLaunchArgument("start_nav2", default_value="true"),
            DeclareLaunchArgument("start_rviz", default_value="true"),
            DeclareLaunchArgument("rviz_config", default_value="nav2_default"),
            DeclareLaunchArgument("slam_resolution", default_value="0.05"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
