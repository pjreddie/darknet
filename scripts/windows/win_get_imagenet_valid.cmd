echo Run install_cygwin.cmd before:

rem http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads
rem https://github.com/amd/OpenCL-caffe/wiki/Instructions-to-create-ImageNet-2012-data


c:\cygwin64\bin\bash -l -c "cd %CD:\=/%/; echo $PWD"


c:\cygwin64\bin\wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_bbox_val_v3.tgz

c:\cygwin64\bin\gzip -d "%CD:\=/%/ILSVRC2012_bbox_val_v3.tgz"

c:\cygwin64\bin\tar --force-local -xvf "%CD:\=/%/ILSVRC2012_bbox_val_v3.tar"


c:\cygwin64\bin\wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar

c:\cygwin64\bin\mkdir -p "%CD:\=/%/imgs"

c:\cygwin64\bin\tar --force-local -xf "%CD:\=/%/ILSVRC2012_img_val.tar" -C "%CD:\=/%/imgs"


echo Wait a few hours...

rem c:\cygwin64\bin\wget https://pjreddie.com/media/files/imagenet_label.sh

c:\cygwin64\bin\dos2unix "%CD:\=/%/windows_imagenet_label.sh"

c:\cygwin64\bin\bash -l -c "cd %CD:\=/%/; %CD:\=/%/windows_imagenet_label.sh"

c:\cygwin64\bin\find "%CD:\=/%/labelled" -name \*.JPEG > inet.val.list



pause