#!/usr/bin/env pwsh

$url = 'https://developer.download.nvidia.com/compute/cuda/11.5.0/network_installers/cuda_11.5.0_win10_network.exe'

$CudaFeatures = 'nvcc_11.5 cuobjdump_11.5 nvprune_11.5 cupti_11.5 memcheck_11.5 nvdisasm_11.5 nvprof_11.5 ' + `
 'visual_studio_integration_11.5 visual_profiler_11.5 visual_profiler_11.5 cublas_11.5 cublas_dev_11.5 ' + `
 'cudart_11.5 cufft_11.5 cufft_dev_11.5 curand_11.5 curand_dev_11.5 cusolver_11.5 cusolver_dev_11.5 ' + `
 'cusparse_11.5 cusparse_dev_11.5 npp_11.5 npp_dev_11.5 nvrtc_11.5 nvrtc_dev_11.5 nvml_dev_11.5 ' + `
 'occupancy_calculator_11.5 '

try {
  Write-Host 'Downloading CUDA...'
  Invoke-WebRequest -Uri $url -OutFile "cuda_11.5.0_win10_network.exe"
  Write-Host 'Installing CUDA...'
  $proc = Start-Process -PassThru -FilePath "./cuda_11.5.0_win10_network.exe" -ArgumentList @('-s ' + $CudaFeatures)
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
