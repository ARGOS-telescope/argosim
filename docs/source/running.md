# Execution

## Graphical User Interface (GUI)
Argosim provides a graphical user interface (GUI) for simulating radio interferometric observations.
Running the GUI requires the `pyQt6` package, which can be installed with the following command:
```
pip install argosim[gui]
```
For opening the GUI, run the following command:
```
python app/argosim-gui.py
```

## Test example on the container
```
(argosim) root@container_id:/workdir# python /home/scripts/test.py
```
The output images are saved to `/workdir` inside the container and will be available in the host machine at `$PWD` (the directory from where the container was run).

## Launch a jupyter notebook server
```
docker run -p 8888:8888 -v $PWD:/workdir --rm ghcr.io/argos-telescope/argosim:main notebook
```
The jupyter notebook server is running on the container. Copy the url (127.0.0.1:8888/tree?token=...) and paste it on the browser.

To exit the container, shut down the jupyter kernel from the browser. The container will be automatically stopped.

All the notebooks created or modified in the container will be saved in the host machine at `$PWD`. If wanted to run an existing notebook, copy it to the current mounted directory (`$PWD`) before or while runing the container.