## enable or disable installed components

$install_cuda=$false
$vcpkg_folder=".\"
$temp_folder=".\temp"

###########################

New-Item -Path . -Name $temp_folder -ItemType "directory"
Set-Location $temp_folder

# Download and install Chocolatey
Set-ExecutionPolicy unrestricted
Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
choco.exe install -y cmake ninja powershell git vscode
choco-exe install -y visualstudio2019buildtools --package-parameters "--add Microsoft.VisualStudio.Component.VC.CoreBuildTools --includeRecommended --includeOptional --passive --locale en-US --lang en-US"

if ($install_cuda) {
  # Download and install CUDA
  Invoke-WebRequest https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.96_win10.exe -OutFile .\cuda_setup.exe
  .\cuda_setup.exe -s nvcc_10.1 cuobjdump_10.1 nvprune_10.1 cupti_10.1 gpu_library_advisor_10.1 memcheck_10.1 nvdisasm_10.1 nvprof_10.1 visual_profiler_10.1 visual_studio_integration_10.1 cublas_10.1 cublas_dev_10.1 cudart_10.1 cufft_10.1 cufft_dev_10.1 curand_10.1 curand_dev_10.1 cusolver_10.1 cusolver_dev_10.1 cusparse_10.1 cusparse_dev_10.1 nvgraph_10.1 nvgraph_dev_10.1 npp_10.1 npp_dev_10.1 nvrtc_10.1 nvrtc_dev_10.1 nvml_dev_10.1 occupancy_calculator_10.1 fortran_examples_10.1

  $env:CUDA_PATH = "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA\v10.1"
  $env:CUDA_PATH_V10_1 = $env:CUDA_PATH
  $env:CUDA_TOOLKIT_ROOT_DIR = $env:CUDA_PATH
  $env:PATH += ";${env:CUDA_PATH}\bin;"

  $features = "full"
}
else {
  $features = "opencv-base,weights,weights-train"
}

Remove-Item -r $temp_folder
Set-Location ..
Set-Location $vcpkg_folder\
git.exe clone https://github.com/Microsoft/vcpkg
Set-Location vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg.exe install darknet[${features}]:x64-windows
