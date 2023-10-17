from setuptools import setup, find_packages
import subprocess
import sys
import os

__env_name__ = "uMAIA"

# Attempt to create a Conda environment from a .yml file
try:
    subprocess.run(["conda", "env", "create", "-f", "environment.yml", "--name", __env_name__])
    subprocess.run(["conda", "run", "-n", __env_name__, "pip", "install", "--force-reinstall", "jax[cuda12_pip]", "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"], check=True)
except subprocess.CalledProcessError:
    print("Failed to create conda environment from environment.yml.")
    sys.exit(1)
    
    
# Attempt to install a specific package via pip (force install)
try:
    subprocess.run(["conda", "run", "-n", __env_name__, "pip", "install", "--force-reinstall", "numpy==1.25.2"], check=True)
except subprocess.CalledProcessError:
    print("Failed to force-install the pip package.")
    sys.exit(1)
    
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

__version__ = 0.1
setup(
    name='MAIA',
    version= __version__,
    packages=find_packages(),
    description='Toolbox for the processing and analysis of MALDI-MSI data',
    author='@lamanno-epfl',
    author_email='gioele.lamanno@epfl.ch',
    install_requires=[
        # Your pip package requirements here
    ],
    url="https://github.com/lamanno-epfl/uMAIA"
)