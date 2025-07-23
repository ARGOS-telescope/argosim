# Installation 

## Basic installation
```
pip install argosim
```

## Local installation
```
git clone https://github.com/ARGOS-telescope/argosim.git
cd argosim
pip install .
```

## Docker installation

### Build the docker image (optional)
```
cd argosim
docker build -t ghcr.io/argos-telescope/argosim:main .
```
Build the doker image from the argosim repository directory. The image will be tagged as `ghcr.io/argos-telescope/argosim:main`.
This step is optional if you want to use the pre-built image from the GitHub container registry.
You can skip this step and go directly to the next section to pull the image from the registry.

### Pull the docker image
```
docker pull ghcr.io/argos-telescope/argosim:main
```
Directly pull the docker image from the github container registry. You may need to login to the registry before pulling the image.

### Run a Docker container
```
docker run -itv $PWD:/workdir --rm ghcr.io/argos-telescope/argosim:main
```
Run the an _argosim_ container with an interactive shell. Mount the current directory (`$PWD`) to the container's workdir. 
The modifications and outputs produced while running in the container will be saved in the host machine at `$PWD`.
The argosim files (src, scripts, notebooks, etc.) are located at the `/home` directory in the container.
