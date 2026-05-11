from glob import glob
import os

from setuptools import setup


package_name = "rl_explore_gazebo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "maps"), glob("maps/*.npy")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.sdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="zbf",
    maintainer_email="zbf@example.com",
    description="Gazebo worlds and launch files for RL exploration.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "generate_world = rl_explore_gazebo.generate_world:main",
            "odom_tf_publisher = rl_explore_gazebo.odom_tf_publisher:main",
        ],
    },
)
