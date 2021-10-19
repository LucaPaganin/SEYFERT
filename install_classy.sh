CLASSURL="https://lesgourg.github.io/class_public/class_public-3.0.1.tar.gz"
CLASSDIRNAME=$(echo $(basename ${CLASSURL}) | sed "s/.tar.gz//")

wget $CLASSURL
tar -zvxf "${CLASSDIRNAME}.tar.gz"
rm -r "${CLASSDIRNAME}.tar.gz"
cd ${CLASSDIRNAME}
make
cd python
python3 setup.py install --user
