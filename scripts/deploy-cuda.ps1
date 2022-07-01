#!/usr/bin/env pwsh

param (
  [switch]$DisableVisualStudioFeatures = $false,
  [switch]$DisableSilentMode = $false
)

Import-Module -Name $PSScriptRoot/utils.psm1 -Force

$url = "https://developer.download.nvidia.com/compute/cuda/${cuda_version_full}/network_installers/cuda_${cuda_version_full}_windows_network.exe"

$CudaFeatures = "nvcc_${cuda_version_short} cuobjdump_${cuda_version_short} nvprune_${cuda_version_short} " + `
  " cupti_${cuda_version_short} memcheck_${cuda_version_short} nvdisasm_${cuda_version_short} nvprof_${cuda_version_short} " + `
  " cublas_${cuda_version_short} cublas_dev_${cuda_version_short} nvjpeg_${cuda_version_short} nvjpeg_dev_${cuda_version_short} " + `
  " nvtx_${cuda_version_short} cuxxfilt_${cuda_version_short} sanitizer_${cuda_version_short} " + `
  " cudart_${cuda_version_short} cufft_${cuda_version_short} cufft_dev_${cuda_version_short} curand_${cuda_version_short} " + `
  " curand_dev_${cuda_version_short} cusolver_${cuda_version_short} cusolver_dev_${cuda_version_short} " + `
  " cusparse_${cuda_version_short} cusparse_dev_${cuda_version_short} npp_${cuda_version_short} npp_dev_${cuda_version_short} " + `
  " nvrtc_${cuda_version_short} nvrtc_dev_${cuda_version_short} nvml_dev_${cuda_version_short} " + `
  " occupancy_calculator_${cuda_version_short} documentation_${cuda_version_short} "

if (-Not $DisableVisualStudioFeatures) {
  $CudaFeatures = $CudaFeatures + "visual_studio_integration_${cuda_version_short} visual_profiler_${cuda_version_short}  "
}

if ($DisableSilentMode) {
  $SilentFlag = ' '
}
else {
  $SilentFlag = '-s '
}

try {
  Push-Location $PSScriptRoot
  Write-Host "Downloading CUDA from $url..."
  Invoke-WebRequest -Uri $url -OutFile "cuda_${cuda_version_full}_windows_network.exe"
  Write-Host 'Installing CUDA...'
  $proc = Start-Process -PassThru -FilePath "./cuda_${cuda_version_full}_windows_network.exe" -ArgumentList @($SilentFlag + $CudaFeatures)
  $proc.WaitForExit()
  $exitCode = $proc.ExitCode
  Pop-Location
  if ($exitCode -eq 0) {
    Write-Host 'Installation successful!'
  }
  else {
    Throw "Installation failed! Exited with $exitCode."
  }
}
catch {
  Throw "Failed to install CUDA! $($_.Exception.Message)"
}
