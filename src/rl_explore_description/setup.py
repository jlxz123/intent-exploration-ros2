from glob import glob
import os

from setuptools import setup


package_name = "rl_explore_description"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*.xacro")),
        (os.path.join("share", package_name, "urdf", "sensors"), glob("urdf/sensors/*.xacro")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="zbf",
    maintainer_email="zbf@example.com",
    description="Robot description resources for the RL exploration Gazebo demo.",
    license="MIT",
    tests_require=["pytest"],
)
