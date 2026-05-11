from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("intent_topic", default_value="/rl_explore/intent_direction"),
            Node(
                package="rl_explore_policy",
                executable="intent_node",
                name="rl_explore_intent_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "intent_topic": LaunchConfiguration("intent_topic"),
                    }
                ],
            ),
        ]
    )
