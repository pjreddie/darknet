echo Run install_cygwin.cmd before:

rem http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html

c:\cygwin64\bin\bash -l -c "cd %CD:\=/%/; echo $PWD"


c:\cygwin64\bin\wget http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Human3.zip

c:\cygwin64\bin\unzip -o "%CD:\=/%/Human3.zip"



c:\cygwin64\bin\dos2unix "%CD:\=/%/windows_otb_get_labels.sh"

c:\cygwin64\bin\bash -l -c "cd %CD:\=/%/; %CD:\=/%/windows_otb_get_labels.sh"

rem c:\cygwin64\bin\find "%CD:\=/%/img" -name \*.jpg > otb_train.txt



pause