from setuptools import setup, find_packages

setup(
    name="aloha_env",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gym",
        "numpy",
        "opencv-python",
        # Assumes frankateach is available in the environment for ZMQ comms
    ],
    description="Aloha robot gym environment compatible with FrankaEnv API",
)

