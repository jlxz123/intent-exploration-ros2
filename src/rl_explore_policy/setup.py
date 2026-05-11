from glob import glob
import os

from setuptools import setup


package_name = "rl_explore_policy"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "checkpoints"), glob("checkpoints/*.pth")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="zbf",
    maintainer_email="zbf@example.com",
    description="CCRL policy inference node for ROS 2 and Gazebo.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "policy_node = rl_explore_policy.policy_node:main",
            "intent_node = rl_explore_policy.intent_node:main",
            "hand_intent_node = rl_explore_policy.hand_intent_node:main",
        ],
    },
)
