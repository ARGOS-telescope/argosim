# ARGOSIM tutorial - Part 0
Setting up the environment and installing the ARGOSIM package.

`argosim` is a Python package for radio interferometric simulations. It can simulate observations for multiple array configurations and observation strategies. This tutorial will guide you through the installation process and initial setup.

## Local Installation
The package can be installed locally using the `pip` package manager. You will require Python 3.11 or higher to install the package. You can install the package using the following command: 

```bash
pip install argosim
```

### Installing in a conda environment
You may want to install `argosim` and other packages required for this tutorial in a conda environment (mini-conda, anaconda, etc.). You can create a conda environment using the following command:

```bash
conda create -n argosim python=3.11
conda activate argosim
```
then install the package using `pip` as mentioned above.

Useful conda commands:
```bash
# List all conda environments
conda env list
# Deactivate the current environment
conda deactivate
# Remove a conda environment
conda env remove -n <env_name>
```

### Installing from source
If you want to contribute to the development of the package, you can install the package from the source code available on the [GitHub repository](https://github.com/ARGOS-telescope/argosim) with the following command:

```bash
git clone https://github.com/ARGOS-telescope/argosim.git
cd argosim
pip install .
```

For contributing through GitHub, you might want to clone using SSH instead of HTTPS for easier authentication. Use the following command:

```bash
git clone git@github.com:ARGOS-telescope/argosim.git
```

## Docker installation
You can also use the ARGOSIM package through a Docker container. The Docker [image](https://github.com/ARGOS-telescope/argosim/pkgs/container/argosim) is available on the GitHub docker registry. You can pull the image using the following command:

```bash
docker pull ghcr.io/argos-telescope/argosim:main
```

Then you can run the container using the following command:

```bash
docker run -itv $PWD:/workdir --rm ghcr.io/argos-telescope/argosim:main
```

This will run an argosim container with an interactive shell. Mount the current directory (`$PWD`) to the container’s `/workdir`. The modifications and outputs produced while running in the container will be saved in the host machine at `$PWD`. The argosim files (src, scripts, notebooks, etc.) are located at the `/home` directory in the container.

The options used in the [`docker run`](https://docs.docker.com/reference/cli/docker/container/run/) command are:
- `-i` for running the container in interactive mode.
- `-t` for tty mode.
- `-v` to bind mount a volume (host directory to container directory). In this case, the current directory (`$PWD`) is mounted to the `/workdir` directory in the container.
- `--rm` to remove the container after exiting.

## Testing the installation

You can test the installation by importing the package in a Python script or a Jupyter notebook. Run the following code snippet to check if the package is installed correctly:

```python
import argosim
help(argosim)
```

If the package is installed correctly, you will see the help message for the `argosim` package.

```python
Help on package argosim:

NAME
    argosim - ARGOSIM.

PACKAGE CONTENTS
    antenna_utils
    beam_utils
    clean
    data_utils
    imaging_utils
    metrics_utils
    plot_utils
    rand_utils

FILE
    /path/to/argosim/__init__.py
```