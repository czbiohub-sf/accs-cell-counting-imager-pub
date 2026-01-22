# cell-counting-imager
This is the software that runs the Cell Counting Imager, an imaging based automated cell counting instrument used with the Automated Cell Culture Splitter (ACCS). The release of this software accompanies the 2024 article "Open-source cell culture automation system with integrated cell counting for passaging microplate cultures". Refer to the article for more context.

Links to the preprint and other Automated Cell Culture Splitter software components and documentation can be found on the [main repository](https://github.com/czbiohub-sf/2024-accs-pub).

## Requirements
Refer to `pyproject.toml` for Python dependencies.

## Installation
The CCI software should be installed in a dedicated Python virtual environment. The exact procedure for doing so depends on which tools are popular this week, but for example:
```
pipx git+https://github.com/czbiohub-sf/accs-cell-counting-imager-pub.git@main
```
After installing the package, run `cci_config` to initialize the local configuration file if this is a new setup.

## Usage
The following commands are available:

`cci_config` creates a local configuration file from a template and tells you where it is.
```
cci_config [-h] [--config {cci25}] [--force] [--debug]

initialize the local configuration file for the CCI

options:
  -h, --help        show this help message and exit
  --config {cci25}  which base instrument config to use (default: cci25)
  --force           overwrite local config file if it exists
  --debug           print debug messages
```

`cci_test` can be used to test whether the stage and camera are connected and working properly.
```
usage: cci_test [-h] [--config NAME] [--debug] [--stage PORTNAME | --stage-auto] [--camera]

options:
  -h, --help        show this help message and exit
  --config NAME     specify the base hardware configuration to use (default: cci25)
  --debug           print debug messages

Stage test:
  Select one of these options to run a test routine to verify the CCI stage is functioning properly.

  --stage PORTNAME  Connect to the serial port identified by PORTNAME (e.g. COM3, /dev/ttyUSB0, etc.)
  --stage-auto      Attempt to automatically discover which port the CCI is connected to baed on the USB VID/PID of
                    the serial adapter cable. In case of ambiguity the program will print the names of all candidates.

Camera:
  Select this option to verify that the software can communicate with the CCI camera.

  --camera          Attempt to connect to the camera and grab an image.
```


`cci_run` is the program responsible for primary functionality of the instrument.
```
usage: cci_run [-h] [--config {cci1,cci25_sim,cci25}] [--local-config PATH] [--output-dir PATH] [--debug] [--console]
               [--load-bg PATH_LIST] [--recover]

options:
  -h, --help            show this help message and exit
  --config {cci1,cci25_sim,cci25}
                        load builtin config named CONFIG_NAME (stackable)
  --local-config PATH   load config from file at PATH instead of the standard location (stackable)
  --output-dir PATH     save outputs to directory at PATH
  --debug               print debug messages
  --console             open an interactive Python console instead of running the server
  --load-bg PATH_LIST   pre-load a set of background images, where PATH_LIST is a JSON array of paths
  --recover             attempt to resume using the most recent CSV file and background images
```


## Maintainers
This repository is currently maintained by [greg.courville@biohub.org](mailto:greg.courville@biohub.org).

