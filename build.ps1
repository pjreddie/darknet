#!/usr/bin/env pwsh

$number_of_build_workers=8
#$shared_lib="-DBUILD_SHARED_LIBS:BOOL=ON"
$force_using_include_libs=$false

#$my_cuda_compute_model=75    #Compute capability for Tesla T4, RTX 2080
#$my_cuda_compute_model=72    #Compute capability for Jetson Xavier
#$my_cuda_compute_model=70    #Compute capability for Tesla V100
#$my_cuda_compute_model=62    #Compute capability for Jetson TX2
#$my_cuda_compute_model=61    #Compute capability for Tesla P40
#$my_cuda_compute_model=60    #Compute capability for Tesla P100
#$my_cuda_compute_model=53    #Compute capability for Jetson TX1
#$my_cuda_compute_model=52    #Compute capability for Tesla M40/M60
#$my_cuda_compute_model=37    #Compute capability for Tesla K80
#$my_cuda_compute_model=35    #Compute capability for Tesla K20/K40
#$my_cuda_compute_model=30    #Compute capability for Tesla K10, Quadro K4000

if ((Test-Path env:VCPKG_ROOT) -and -not $force_using_include_libs) {
  $vcpkg_path = "$env:VCPKG_ROOT"
  Write-Host "Found vcpkg in VCPKG_ROOT: $vcpkg_path"
}
elseif ((Test-Path "${env:WORKSPACE}\vcpkg") -and -not $force_using_include_libs) {
  $vcpkg_path = "${env:WORKSPACE}\vcpkg"
  Write-Host "Found vcpkg in WORKSPACE\vcpkg: $vcpkg_path"
}
else {
  Write-Host "Skipping vcpkg-enabled builds because the VCPKG_ROOT environment variable is not defined, using self-distributed libs`n" -ForegroundColor Yellow
}

if ($null -eq $env:VCPKG_DEFAULT_TRIPLET) {
  Write-Host "No default triplet has been set-up for vcpkg. Defaulting to x64-windows`n" -ForegroundColor Yellow
  $vcpkg_triplet = "x64-windows"
}
else {
  $vcpkg_triplet = $env:VCPKG_DEFAULT_TRIPLET
}

if ($vcpkg_triplet -Match "x86") {
  Throw "darknet is supported only in x64 builds!"
}

if ($null -eq (Get-Command "cl.exe" -ErrorAction SilentlyContinue)) {
  $vstype = "Professional"
  if (Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\2017\${vstype}\Common7\Tools") {
  }
  else {
    $vstype = "Enterprise"
    if (Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\2017\${vstype}\Common7\Tools") {
    }
    else {
      $vstype = "Community"
    }
  }
  Write-Host "Found VS 2017 ${vstype}"
  Push-Location "C:\Program Files (x86)\Microsoft Visual Studio\2017\${vstype}\Common7\Tools"
  cmd /c "VsDevCmd.bat -arch=x64 & set" |
    ForEach-Object {
    if ($_ -match "=") {
      $v = $_.split("="); set-item -force -path "ENV:\$($v[0])"  -value "$($v[1])"
    }
  }
  Pop-Location
  Write-Host "Visual Studio 2017 ${vstype} Command Prompt variables set.`n" -ForegroundColor Yellow
}

if ($null -eq (Get-Command "nvcc.exe" -ErrorAction SilentlyContinue)) {
  if (Test-Path env:CUDA_PATH) {
    $env:PATH += ";${env:CUDA_PATH}\bin"
  }
  else {
    Write-Host "Unable to find CUDA, if necessary please install it or define a CUDA_PATH env variable pointing to the install folder`n" -ForegroundColor Yellow
  }
}

if (Test-Path env:CUDA_PATH) {
  if (-Not(Test-Path env:CUDA_TOOLKIT_ROOT_DIR)) {
    $env:CUDA_TOOLKIT_ROOT_DIR = "${env:CUDA_PATH}"
    Write-Host "Added missing env variable CUDA_TOOLKIT_ROOT_DIR`n" -ForegroundColor Yellow
  }
}

if($my_cuda_compute_model) {
  $additional_build_setup = "-DCUDA_COMPUTE_MODEL=${my_cuda_compute_model}"
}

if ($vcpkg_path) {
  # DEBUG
  New-Item -Path .\build_win_debug -ItemType directory -Force
  Set-Location build_win_debug
  cmake -G "Visual Studio 15 2017" -T "host=x64" -A "x64" "-DCMAKE_TOOLCHAIN_FILE=$vcpkg_path\scripts\buildsystems\vcpkg.cmake" "-DVCPKG_TARGET_TRIPLET=$vcpkg_triplet" "-DCMAKE_BUILD_TYPE=Debug" $shared_lib $additional_build_setup ..
  cmake --build . --config Debug --parallel ${number_of_build_workers} --target install
  Remove-Item DarknetConfig.cmake
  Remove-Item DarknetConfigVersion.cmake
  Set-Location ..

  # RELEASE
  New-Item -Path .\build_win_release -ItemType directory -Force
  Set-Location build_win_release
  cmake -G "Visual Studio 15 2017" -T "host=x64" -A "x64" "-DCMAKE_TOOLCHAIN_FILE=$vcpkg_path\scripts\buildsystems\vcpkg.cmake" "-DVCPKG_TARGET_TRIPLET=$vcpkg_triplet" "-DCMAKE_BUILD_TYPE=Release" $shared_lib $additional_build_setup ..
  cmake --build . --config Release --parallel ${number_of_build_workers} --target install
  Remove-Item DarknetConfig.cmake
  Remove-Item DarknetConfigVersion.cmake
  Copy-Item *.dll ..
  Set-Location ..
}
else {
  # USE LOCAL PTHREAD LIB AND LOCAL STB HEADER, NO VCPKG, ONLY RELEASE MODE SUPPORTED
  # if you want to manually force this case, remove VCPKG_ROOT env variable and remember to use "vcpkg integrate remove" in case you had enabled user-wide vcpkg integration
  New-Item -Path .\build_win_release_novcpkg -ItemType directory -Force
  Set-Location build_win_release_novcpkg
  cmake -G "Visual Studio 15 2017" -T "host=x64" -A "x64" $shared_lib $additional_build_setup ..
  cmake --build . --config Release --parallel ${number_of_build_workers} --target install
  Remove-Item DarknetConfig.cmake
  Remove-Item DarknetConfigVersion.cmake
  Copy-Item ..\3rdparty\pthreads\bin\pthreadVC2.dll ..
  Set-Location ..
}
