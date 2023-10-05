# uMAIA
Toolbox for the processing and analysis of MALDI-MSI data


![alt text](figs/MAIA%20workflow.png)


## Installation

To begin, install the environment by navigating to the /uMAIA directory and running `python3 setup.py`.


### Gurobi
The molecule matching portion of the pipeline requires Gurobi. You will need to (1) install the Gurobi installer and (2) activate a license that is free for those are affiliated with an academic institution. Concerning (1), download the archive with 

`wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz` to the directory you want to install it in.

Unpack the archive with

`tar -xzvf gurobi9.1.2_linux64.tar.gz`

Activate the license with

`gurobi912/linux64/bin/grbgetkey <licensekey>`



### GPU usage restrictions
If you intend on applying uMAIA's normalisation algorithm on a GPU, please note that JAX will automatically use between 75 and 90 % of available GPU memory. To indicate that you would only like to allocate memory that is needed, you may export an environment variable after activating the environment with `export XLA_PYTHON_CLIENT_PREALLOCATE=false`


