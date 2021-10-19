#!/bin/bash

SETUP_CMD=${1:-install}
VENV_NAME="venv_seyfert"

VENV_PATH="${HOME}/.venv/${VENV_NAME}"

pip3 install virtualenv
if [ ! -e "${VENV_NAME}" ]; then
  echo "Creating virtual environment ${VENV_PATH}"
  mkdir -p "${HOME}/.venv"
  virtualenv -p python3 "${VENV_PATH}"
else
  echo "Virtual environment ${VENV_NAME} already exists, skipping its creation"
fi

echo "Activating virtual environment ${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
echo "Upgrading pip"
pip3 install --upgrade pip
echo "Installing requirements"
pip3 install -r requirements.txt
echo "Upgrading setuptools and wheel"
python3 -m pip3 install --upgrade setuptools wheel
echo "Installing ipython ipykernel"
pip3 install ipython ipykernel
echo "Installing SEYFERT"
python3 setup.py "${SETUP_CMD}"

echo "Deactivating virtual environment ${VENV_PATH}"
deactivate
echo "Done"

