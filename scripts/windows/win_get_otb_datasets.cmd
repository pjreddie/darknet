echo Run install_cygwin.cmd before:

rem http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html

c:\cygwin64\bin\bash -l -c "cd %CD:\=/%/; echo $PWD"

c:\cygwin64\bin\dos2unix "%CD:\=/%/otb_get_labels.sh"


c:\cygwin64\bin\bash -l -c "cd %CD:\=/%/; %CD:\=/%/otb_get_labels.sh Suv 320 240"

c:\cygwin64\bin\bash -l -c "cd %CD:\=/%/; %CD:\=/%/otb_get_labels.sh Liquor 640 480"

c:\cygwin64\bin\bash -l -c "cd %CD:\=/%/; %CD:\=/%/otb_get_labels.sh Freeman4 360 240"

c:\cygwin64\bin\bash -l -c "cd %CD:\=/%/; %CD:\=/%/otb_get_labels.sh Human3 480 640"

pause