from setuptools import find_packages, setup

package_name = "legged_rl_env"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Grigorii",
    maintainer_email="sizgrisha@gmail.com",
    description="Gymnasium custom environment implementation",
    license="Apache 2.0",
    extras_require={
        "test": [],
    },
    entry_points={
        "console_scripts": [
            "go2_env = legged_rl_env.go2_env:main",
            "go2_node = legged_rl_env.go2_node:main",
        ],
    },
)
