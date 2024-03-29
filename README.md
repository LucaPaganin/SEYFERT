# SEYFERT

SurvEY FishEr foRecast Tool

## Installation
You have two possibilities to install and run SEYFERT:
* running on your machine in a python virtual environment (venv);
* running on a docker container.

You can also run it directly on your machine without creating a venv, but this is not suggested, as you will install all
the python package dependencies in your base environment. However you may also decide that you don't care and install it 
directly on your system, if you know what you are doing.

### Installing in a virtual environment
#### Prerequisites
The mandatory prerequisites in this case are:
* `Python >=3.5`: SEYFERT is written completely in Python3, in particular Python >= 3.5 is required. A suggested way to 
  install python3 is to download the Anaconda release (see for example https://www.anaconda.com/products/individual)
* `gfortran`: it is needed to install `CAMB`. To install it on Mac OS consider to use Homebrew (see https://brew.sh/). 
  After downloading and installing homebrew, run `brew install gfortran`.
* `virtualenv`: it is needed for installing the code with the `installer.sh` script, described in the following. 

#### Installation procedure
In this case you can run the `installer.sh` script, which is ready-to-use as it is. This script will create a new virtual 
environment at the path `$HOME/.venv/venv_seyfert` and it will activate it. Then it will install the requirements in the 
virtual environment just created, and at the end it will install SEYFERT with the command `python setup.py install`.
After the installation a bunch of command line python scripts will be installed and made available from everywhere. These
are the scripts containe in the `bin` folder of the repository.

##### Optional: install CLASS Boltzmann code
You can optionally install `CLASS`, more precisely its python wrapper `classy`, which can be used by SEYFERT to compute 
the matter power spectrum alternatively to `CAMB`, which is the default Boltzmann code. The script to install `CLASS` is 
`seyfert/install_classy.sh`. If you want to install classy, just run this script, but pay attention because it is not guaranteed
to work on your system, since it has external dependencies such as `gcc`. For more information on this refer to

https://lesgourg.github.io/class_public/class.html

### Running on a docker container
#### Prerequisites
This procedure has tha advantage that you will run the code in a *completely isolated system environment*, not only from
the python packages point of view, but for all the system and external libraries that you need for running the code. The
main disadvantage is that Docker is not so lightweight, and may consume a discrete amount of resources. However you can 
limit these by configuring Docker itself. You should also need to install docker on your machine, for which we do not report
all the instructions, since are already described here

https://www.docker.com/products/docker-desktop

The installation and configuration of Docker depend on your particular system, but on MacOS it is pretty straightforward,
as well as in Linux. For Windows there will be some additional steps to perform (such as enabling Hyper-V), but you will find
all the relevant instruction in the above link. You are not required to know any docker syntax or command, since in the 
following we report all the commands that are needed to be executed.

#### Pulling the image
You can retrieve the official docker image for running SEYFERT via you INFN credentials. The steps are the following:
* in your terminal execute the command  
```
docker login baltig.infn.it:4567
```
and authenticate with your INFN credentials.
* then run `docker pull baltig.infn.it:4567/euclid-ge/seyfert`, and this will start the download procedure. It could take 
  a few minutes, so please be patient.
After that you will be ready to create a docker container for running SEYFERT inside it.

#### Creating an interactive container
Now that you have downloaded the docker image, you are ready to create your container. In order to do so run you will have
to use the `docker run` command. The basic syntax of docker run is

```
docker run [<OPTIONS>] <IMAGE_NAME> <COMMAND>
```

This creates a container (instance) from an image named `<IMAGE_NAME>`, which you will need to downloaded or to be available
on DockerHub. In the case of SEYFERT you will have to download it manually as described in the previous section, since the
image is not available through DockerHub. The particular command we suggest to run to create the container is the following

```
docker run -it --rm -p 8080:8080 -v $HOME:/home baltig.infn.it:4567/euclid-ge/seyfert /bin/bash
```

In detail:
* `-it`: this is the to run the container interactively;
* `--rm`: it serves to remove the container after closing it;
* `-p 8080:8080`: it shares the port `8080` of the host machine with the port `8080` of the container. It is needed for 
  running for example jupyter interactively inside the container by opening it into your browser.
* `-v $HOME:/home`: it shares your home directory with the home directory of the container. This is needed to persist 
  the outputs you obtain after having closed the temporary container. For this reason it is important to *save your 
  outputs in a folder which is shared with the host*. Otherwise you won't be able to access the output data from outside 
  the container, i.e. from your computer (the host). Saving the output in a sub-folder of your home directory will be 
  sufficient.
* `baltig.infn.it:4567/euclid-ge/seyfert`: it is the image name.
* `/bin/bash`: it is the command you will run inside the container, which is a simple `BASH` shell.

After that you will have opened an interactive shell in a brand new running container. Now go into the path on which you
have downloaded SEYFERT, and install it by running `python3 setup.py install`. 

#### Opening jupyter inside the container
If you want you can run SEYFERT from jupyter inside the container. In that case you can open `jupyter-lab` on the container
by running the command
```
jupyter-lab --no-browser --port=8080 --allow-root --ip=0.0.0.0
```
Then you should see in the terminal some output like 
```
http://127.0.0.1:8080/lab?token=813c70c730dfc8f5577f4d33ec8b196739b202f1962e76d1
```
The part of the URL following `token=` will differ, since it is your access token for that particular jupyter-lab server. 
Simply copy the full link and paste it into the browser URL bar. You
should now have a running jupyter-lab server on your container, from which you can run python commands, including 
SEYFERT modules and functions.

## Running a forecast
In SEYFERT the steps for performing a forecast are essentially four:
* Computation of the matter power spectra for all the necessary cosmologies (varied w.r.t. reference cosmology);
* Computation of the angular power spectra for all cosmologies, including nuisance parameters variations;
* Computation of the derivatives of the angular power spectra w.r.t. both cosmological and nuisance parameters;
* Computation of the Fisher matrices.

These steps can be executed all at once or separately. In the second case it is necessary 
to tell the code where to find the results of the necessary previous steps that are not 
included in the computation.

### SEYFERT Demo notebooks
Two jupyter notebooks are available in the `seyfert/demo` directory:
* `SeyfertDemo.ipynb`: this is a notebook showing the usage of 
  the main python classes with which the code is written.
* `InteractiveForecast.ipynb`: notebook for running a complete forecast interactively.

### Creating a forecast configuration
SEYFERT has a simple Graphic User Interface (GUI) for creating the configuration of the
forecast. It can be opened by typing `gui_config` into a shell with the virtual environment
active. This will open a graphic window through which all the options can be changed. After 
that click the save button and enter the name (without extension) of the file to which the
configuration will be saved.

### Main entry point
The main script for performing a forecast with SEYFERT is the `job_submitter` script. 
In order to print the usage of the script execute the following command

```
job_submitter -h
```

The mandatory parameters for a run are
  * `--forecast_config (-fcfg)`: the forecast configuration XML file;
  * `--execution (-ex)`: the execution mode for the forecast (see help message 
    for more details);
  * `--input_data_dir (-i)`: the path to the input data directory.
    
Moreover at least one of the 4 following task flags must be specified:
  * `--powerspectrum (-pmm)` for computing the power spectra;
  * `--angular (-cl)` for computing the angular power spectra;
  * `--derivative (-der)` for computing the derivatives of the angular power spectra;
  * `--fisher (-fish)`  for computing the fisher matrices.
