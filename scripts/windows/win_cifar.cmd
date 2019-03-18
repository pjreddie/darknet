echo Run install_cygwin.cmd before:


c:\cygwin64\bin\wget https://pjreddie.com/media/files/cifar.tgz

c:\cygwin64\bin\gzip -d "%CD:\=/%/cifar.tgz"

c:\cygwin64\bin\tar --force-local -xvf "%CD:\=/%/cifar.tar"

c:\cygwin64\bin\cat "%CD:\=/%/labels.txt"


c:\cygwin64\bin\find "%CD:\=/%/cifar/train" -name \*.png > "%CD:\=/%/cifar/train.list"

c:\cygwin64\bin\find "%CD:\=/%/cifar/test" -name \*.png > "%CD:\=/%/cifar/test.list"



pause