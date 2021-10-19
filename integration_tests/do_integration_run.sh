#!/bin/bash

CODEVERSION=$1
EXECUTION=$2

if [ -z "${CODEVERSION}" ] || [ -z "${EXECUTION}" ] ; then
  echo "Missing codeversion or execution mode"
  exit 1
fi

CURBRANCH=$(git branch | egrep "\* [a-zA-Z]*" | sed "s/\* //")

WORKDIR="${SEYFERT_PATH}/integration_tests"
INPUTDATADIR="${SPCPH_PATH}/input_data"

echo "Current branch: ${CURBRANCH}"
echo "Checking out ${CODEVERSION}"
cd "${SEYFERT_PATH}"
git checkout "${CODEVERSION}"
echo "Running ${CODEVERSION}"
python3 setup.py develop
echo "workdir: ${WORKDIR}"
cd "${WORKDIR}"
echo "Running job_submitter"

job_submitter -id "seyfert_run_integration" -i "${INPUTDATADIR}" -pmm -cl -der -fish -ex "${EXECUTION}" --verbose \
-fcfg "${WORKDIR}/config/fcfg.xml" -pmm_cfg "${WORKDIR}/config/pmm_cfg.json" -cl_cfg "${WORKDIR}/config/cl_cfg.json" \
-der_cfg "${WORKDIR}/config/der_cfg.json" -fish_cfg "${WORKDIR}/config/fish_cfg.json" -res_cfg "${WORKDIR}/config/res_cfg.json"

echo "Returning to branch ${CURBRANCH}"
git checkout "${CURBRANCH}"
echo "Done"
