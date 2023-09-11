from setuptools import setup, find_packages
import subprocess
import sys
import os

# Attempt to create a Conda environment from a .yml file
try:
    subprocess.run(["conda", "env", "create", "-f", "environment.yml", "--name", "myenv"])
    subprocess.run(["conda", "run", "-n", "myenv", "pip", "install", "--force-reinstall", "jax[cuda12_pip]", "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"], check=True)
except subprocess.CalledProcessError:
    print("Failed to create conda environment from environment.yml.")
    sys.exit(1)
    
    
# Attempt to install a specific package via pip (force install)
try:
    subprocess.run(["conda", "run", "-n", "myenv", "pip", "install", "--force-reinstall", "numba==0.57.1"], check=True)
    subprocess.run(["conda", "run", "-n", "myenv", "pip", "install", "--force-reinstall", "numpy==1.23.0"], check=True)
except subprocess.CalledProcessError:
    print("Failed to force-install the pip package.")
    sys.exit(1)
    
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

setup(
    name='MAIA',
    version='0.1',
    packages=find_packages(),
    description='Python Distribution Utilities',
    author='@lamanno-epfl',
    author_email='gioele.lamanno@epfl.ch',
    install_requires=[
        # Your pip package requirements here
    ],
)