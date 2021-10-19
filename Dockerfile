FROM ubuntu:latest

LABEL mantainers="marco.bonici@ge.infn.it; luca.paganin@ge.infn.it"
LABEL version="0.0.1"
LABEL description="This Dockerfile create a Docker image where it is possible to run seyfert"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install vim -y \
    && apt-get install tree \
    && apt-get install wget -y \
    && apt-get install git -y
RUN apt-get install apt-utils -y
RUN apt-get upgrade -y
RUN apt-get autoremove -y
RUN apt-get install python3 -y
RUN apt-get install python3-distutils -y
RUN apt-get install python3-pip -y
RUN apt-get install gfortran -y && apt-get install gcc -y
RUN pip3 install -U pip
RUN pip3 install ipython ipykernel jupyterlab
WORKDIR /home
ADD requirements.txt ./
RUN pip3 install -r requirements.txt
ADD install_classy.sh ./
RUN ./install_classy.sh
ARG jupyter_shotcuts_dir=/root/.jupyter/lab/user-settings/@jupyterlab/shortcuts-extension/
RUN mkdir -p $jupyter_shotcuts_dir
ADD shortcuts.jupyterlab-settings $jupyter_shotcuts_dir/
