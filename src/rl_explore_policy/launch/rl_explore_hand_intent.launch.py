from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("intent_topic", default_value="/rl_explore/intent_direction"),
            DeclareLaunchArgument("camera_index", default_value="0"),
            DeclareLaunchArgument(
                "model_path",
                default_value="/home/zbf/achievement/graduate/hand_dir_detect/hand_landmarker.task",
            ),
            DeclareLaunchArgument("mirror_image", default_value="true"),
            DeclareLaunchArgument("show_window", default_value="true"),
            DeclareLaunchArgument("image_topic", default_value=""),
            DeclareLaunchArgument("buffer_size", default_value="5"),
            DeclareLaunchArgument("cooldown_ms", default_value="1000.0"),
            DeclareLaunchArgument("motion_threshold", default_value="0.02"),
            Node(
                package="rl_explore_policy",
                executable="hand_intent_node",
                name="rl_explore_hand_intent_node",
                output="screen",
                parameters=[
                    {
                        "intent_topic": LaunchConfiguration("intent_topic"),
                        "camera_index": LaunchConfiguration("camera_index"),
                        "model_path": LaunchConfiguration("model_path"),
                        "mirror_image": LaunchConfiguration("mirror_image"),
                        "show_window": LaunchConfiguration("show_window"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "buffer_size": LaunchConfiguration("buffer_size"),
                        "cooldown_ms": LaunchConfiguration("cooldown_ms"),
                        "motion_threshold": LaunchConfiguration("motion_threshold"),
                    }
                ],
            ),
        ]
    )
