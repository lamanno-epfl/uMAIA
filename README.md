# uMAIA
Toolbox for the processing and analysis of MALDI-MSI data. For details please see our preprint on [BioRxiv](https://www.biorxiv.org/content/10.1101/2024.08.20.608739v2)

![alt text](figs/uMaiaLogo.png)


## Installation

Installation has been tested on Linux and MacOS.

To begin, create a new environment with 
`conda create --name uMAIA_env python==3.11.3`

activate the environment with
`conda activate uMAIA_env`

pip install the following packages, in the precise order:

`pip install -U pip==25.1.1 setuptools==80.9.0 wheel==0.45.1`

`pip install --only-binary=:all: --no-cache-dir numcodecs==0.12.0`

`pip install py-cpuinfo==9.0.0`

`pip install cython==3.1.2`

`pip install -U --no-deps scikit-image==0.22.0`

`pip install -U --no-deps lazy_loader==0.4`

Next, clone the uMAIA repository with 
`git clone https://github.com/lamanno-epfl/uMAIA.git`
and navigate into the directory with
`cd uMAIA`

Install the uMAIA package in developer mode with the following command:
`pip install --no-build-isolation --no-cache-dir -e .`

Developer mode will allow you to modify source code in the package while still importing the module as normal. In other words you can edit your code in-place, and the changes are immediately reflected when you import or run the package.

### Gurobi
The molecule matching portion of the pipeline requires Gurobi. You will need to (1) install the Gurobi installer and (2) activate a license that is free for those are affiliated with an academic institution. Concerning (1), download the archive with 

Linux:
`wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz` to the directory you want to install it in.
or MacOS:
`https://packages.gurobi.com/9.1/gurobi9.1.2_macos_universal2.pkg`

Unpack the archive with

Linux:
`tar -xzvf gurobi9.1.2_linux64.tar.gz`
or on MacOS, simply double click the .pkg file and walk through the installation instructions

Activate the license with

Linux:
`gurobi912/linux64/bin/grbgetkey <licensekey>`
MacOS:
`grbgetkey <licensekey>`


## Command line tools

The peakcalling module can be run from the command line, given a list of datasets to process sequentially. This module assumes data is organized in the following way:

```bash
.
|-- __init__.py
|-- molecule_matching
|   |-- __init__.py
|   |-- _match.py
|   |-- _moleculematch.py
|   `-- __pycache__
|       |-- __init__.cpython-311.pyc
|       |-- _match.cpython-311.pyc
|       `-- _moleculematch.cpython-311.pyc
|-- normalization
|   |-- _initialize.py
```

Then, open the file `sequential_commands.sh` and edit its contents (specifically the following line) to include the file names you want to process. For example, copy and paste these two lines into the parentheses inside the 'commands' section.

```"python uMAIA/peak_finding/extract_images.py   --path_data 'rawData/'   --name 'acquisition1_name'"```

```"python uMAIA/peak_finding/extract_images.py   --path_data 'rawData/'   --name 'acquisition2_name'"```

You can then run this file with `bash sequential_commands.sh`.
If you wish to run this in the background `nohup bash sequential_commands.sh > output-sequentialcommand.txt &`

The outputted comments will be saved to `output-sequentialcommand.txt` which will allow you to inspect the progress of the commands.

Note that the default parameters to the peak caller have been optimised for an Orbitrap instrument. If you wish to specify parameters use the following flags `--mzresolution`, `--threshold_count`, `--approximate_interval`, `--smoothing`, `--spectrum_range`.

## GPU usage restrictions
If you intend on applying uMAIA's normalisation algorithm on a GPU, please note that JAX will automatically use between 75 and 90 % of available GPU memory. To indicate that you would only like to allocate memory that is needed, you may export an environment variable after activating the environment with `export XLA_PYTHON_CLIENT_PREALLOCATE=false`


