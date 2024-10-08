# uMAIA
Toolbox for the processing and analysis of MALDI-MSI data. For details please see our preprint on [BioRxiv] ([https://www.genome.gov/](https://www.biorxiv.org/content/10.1101/2024.08.20.608739v2))

![alt text](figs/uMaiaLogo.png)


## Installation

To begin, create a new environment with 
`conda create --name uMAIA_env python==3.11.3`

activate the environment with
`conda activate uMAIA_env`

pip install the package
`pip install --no-cache-dir uMAIA==0.2`


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



### GPU usage restrictions
If you intend on applying uMAIA's normalisation algorithm on a GPU, please note that JAX will automatically use between 75 and 90 % of available GPU memory. To indicate that you would only like to allocate memory that is needed, you may export an environment variable after activating the environment with `export XLA_PYTHON_CLIENT_PREALLOCATE=false`


