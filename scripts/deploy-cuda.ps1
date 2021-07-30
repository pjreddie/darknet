#!/usr/bin/env pwsh

$url = 'https://developer.download.nvidia.com/compute/cuda/11.4.0/network_installers/cuda_11.4.0_win10_network.exe'

$CudaFeatures = 'nvcc_11.4 cuobjdump_11.4 nvprune_11.4 cupti_11.4 memcheck_11.4 nvdisasm_11.4 nvprof_11.4 ' + `
 'visual_studio_integration_11.4 visual_profiler_11.4 visual_profiler_11.4 cublas_11.4 cublas_dev_11.4 ' + `
 'cudart_11.4 cufft_11.4 cufft_dev_11.4 curand_11.4 curand_dev_11.4 cusolver_11.4 cusolver_dev_11.4 ' + `
 'cusparse_11.4 cusparse_dev_11.4 npp_11.4 npp_dev_11.4 nvrtc_11.4 nvrtc_dev_11.4 nvml_dev_11.4 ' + `
 'occupancy_calculator_11.4 '

try {
  Write-Host 'Downloading CUDA...'
  Invoke-WebRequest -Uri $url -OutFile "cuda_11.4.0_win10_network.exe"
  Write-Host 'Installing CUDA...'
  $proc = Start-Process -PassThru -FilePath "./cuda_11.4.0_win10_network.exe" -ArgumentList @('-s ' + $CudaFeatures)
  $proc.WaitForExit()
  $exitCode = $proc.ExitCode
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
