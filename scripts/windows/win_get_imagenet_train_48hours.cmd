echo Run install_cygwin.cmd before:

rem http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads
rem https://github.com/amd/OpenCL-caffe/wiki/Instructions-to-create-ImageNet-2012-data


c:\cygwin64\bin\bash -l -c "cd %CD:\=/%/; echo $PWD"

echo Wait several hours...

c:\cygwin64\bin\wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar

c:\cygwin64\bin\mkdir -p "%CD:\=/%/ILSVRC2012_img_train"

c:\cygwin64\bin\tar --force-local -xf "%CD:\=/%/ILSVRC2012_img_train.tar" -C "%CD:\=/%/ILSVRC2012_img_train"



c:\cygwin64\bin\bash -l -c "cd %CD:\=/%/; %CD:\=/%/windows_imagenet_train.sh"

c:\cygwin64\bin\find "%CD:\=/%/ILSVRC2012_img_train" -name \*.JPEG > imagenet1k.train.list



pause