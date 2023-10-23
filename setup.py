from setuptools import setup, find_packages

__version__ = 0.1

# Read requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
    
setup(
    name='MAIA',
    version= __version__,
    packages=find_packages(),
    description='Toolbox for the processing and analysis of MALDI-MSI data',
    author='@lamanno-epfl',
    author_email='gioele.lamanno@epfl.ch',
    install_requires=requirements,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lamanno-epfl/uMAIA"
)