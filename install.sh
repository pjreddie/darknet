SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

mkdir /etc/darknet
cp -R data/ /etc/darknet/
cp libdarknet.so /usr/lib/
ln -s $SCRIPTPATH/darknet /usr/local/bin/darknet
