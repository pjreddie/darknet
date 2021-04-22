#!/usr/bin/env pwsh

param (
  [switch]$DisableInteractive = $false,
  [switch]$EnableCUDA = $false,
  [switch]$EnableCUDNN = $false,
  [switch]$EnableOPENCV = $false,
  [switch]$EnableOPENCV_CUDA = $false,
  [switch]$UseVCPKG = $false,
  [switch]$DoNotUpdateVCPKG = $false,
  [switch]$DoNotUpdateDARKNET = $false,
  [switch]$DoNotDeleteBuildFolder = $false,
  [switch]$DoNotSetupVS = $false,
  [switch]$DoNotUseNinja = $false,
  [switch]$ForceCPP = $false,
  [switch]$ForceStaticLib = $false,
  [switch]$ForceGCC8 = $false
)

if (-Not $DisableInteractive -and -Not $UseVCPKG) {
  $Result = Read-Host "Enable vcpkg to install darknet dependencies (yes/no)"
  if ($Result -eq 'Yes' -or $Result -eq 'Y' -or $Result -eq 'yes' -or $Result -eq 'y') {
    $UseVCPKG = $true
  }
}

if (-Not $DisableInteractive -and -Not $EnableCUDA -and -Not $IsMacOS) {
  $Result = Read-Host "Enable CUDA integration (yes/no)"
  if ($Result -eq 'Yes' -or $Result -eq 'Y' -or $Result -eq 'yes' -or $Result -eq 'y') {
    $EnableCUDA = $true
  }
}

if ($EnableCUDA -and -Not $DisableInteractive -and -Not $EnableCUDNN) {
  $Result = Read-Host "Enable CUDNN optional dependency (yes/no)"
  if ($Result -eq 'Yes' -or $Result -eq 'Y' -or $Result -eq 'yes' -or $Result -eq 'y') {
    $EnableCUDNN = $true
  }
}

if (-Not $DisableInteractive -and -Not $EnableOPENCV) {
  $Result = Read-Host "Enable OpenCV optional dependency (yes/no)"
  if ($Result -eq 'Yes' -or $Result -eq 'Y' -or $Result -eq 'yes' -or $Result -eq 'y') {
    $EnableOPENCV = $true
  }
}

$number_of_build_workers = 8
#$additional_build_setup = " -DCMAKE_CUDA_ARCHITECTURES=30"

if ($IsLinux -or $IsMacOS) {
  $bootstrap_ext = ".sh"
}
elseif ($IsWindows) {
  $bootstrap_ext = ".bat"
}
Write-Host "Native shell script extension: ${bootstrap_ext}"

if (-Not $IsWindows) {
  $DoNotSetupVS = $true
}

if ($ForceStaticLib) {
  Write-Host "Forced CMake to produce a static library"
  $additional_build_setup = " -DBUILD_SHARED_LIBS=OFF "
}

if ($IsLinux -and $ForceGCC8) {
  Write-Host "Manually setting CC and CXX variables to gcc-8 and g++-8"
  $env:CC = "gcc-8"
  $env:CXX = "g++-8"
}

if ($IsWindows -and -Not $env:VCPKG_DEFAULT_TRIPLET) {
  $env:VCPKG_DEFAULT_TRIPLET = "x64-windows"
}

if ($EnableCUDA) {
  if ($IsMacOS) {
    Write-Host "Cannot enable CUDA on macOS" -ForegroundColor Yellow
    $EnableCUDA = $false
  }
  Write-Host "CUDA is enabled"
}
elseif (-Not $IsMacOS) {
  Write-Host "CUDA is disabled, please pass -EnableCUDA to the script to enable"
}

if ($EnableCUDNN) {
  if ($IsMacOS) {
    Write-Host "Cannot enable CUDNN on macOS" -ForegroundColor Yellow
    $EnableCUDNN = $false
  }
  Write-Host "CUDNN is enabled"
}
elseif (-Not $IsMacOS) {
  Write-Host "CUDNN is disabled, please pass -EnableCUDNN to the script to enable"
}

if ($EnableOPENCV) {
  Write-Host "OPENCV is enabled"
}
else {
  Write-Host "OPENCV is disabled, please pass -EnableOPENCV to the script to enable"
}

if ($EnableCUDA -and $EnableOPENCV -and -not $EnableOPENCV_CUDA) {
  Write-Host "OPENCV with CUDA extension is not enabled, you can enable it passing -EnableOPENCV_CUDA"
}
elseif ($EnableOPENCV -and $EnableOPENCV_CUDA -and -not $EnableCUDA) {
  Write-Host "OPENCV with CUDA extension was requested, but CUDA is not enabled, you can enable it passing -EnableCUDA"
  $EnableOPENCV_CUDA = $false
}
elseif ($EnableCUDA -and $EnableOPENCV_CUDA -and -not $EnableOPENCV) {
  Write-Host "OPENCV with CUDA extension was requested, but OPENCV is not enabled, you can enable it passing -EnableOPENCV"
  $EnableOPENCV_CUDA = $false
}
elseif ($EnableOPENCV_CUDA -and -not $EnableCUDA -and -not $EnableOPENCV) {
  Write-Host "OPENCV with CUDA extension was requested, but OPENCV and CUDA are not enabled, you can enable them passing -EnableOPENCV -EnableCUDA"
  $EnableOPENCV_CUDA = $false
}

if ($UseVCPKG) {
  Write-Host "VCPKG is enabled"
  if ($DoNotUpdateVCPKG) {
    Write-Host "VCPKG will not be updated to latest version if found" -ForegroundColor Yellow
  }
  else {
    Write-Host "VCPKG will be updated to latest version if found"
  }
}
else {
  Write-Host "VCPKG is disabled, please pass -UseVCPKG to the script to enable"
}

if ($DoNotSetupVS) {
  Write-Host "VisualStudio integration is disabled"
}
else {
  Write-Host "VisualStudio integration is enabled, please pass -DoNotSetupVS to the script to disable"
}

if ($DoNotUseNinja) {
  Write-Host "Ninja is disabled"
}
else {
  Write-Host "Ninja is enabled, please pass -DoNotUseNinja to the script to disable"
}

if ($ForceCPP) {
  Write-Host "ForceCPP build mode is enabled"
}
else {
  Write-Host "ForceCPP build mode is disabled, please pass -ForceCPP to the script to enable"
}

Push-Location $PSScriptRoot

$GIT_EXE = Get-Command git 2> $null | Select-Object -ExpandProperty Definition
if (-Not $GIT_EXE) {
  throw "Could not find git, please install it"
}
else {
  Write-Host "Using git from ${GIT_EXE}"
}

if ((Test-Path "$PSScriptRoot/.git") -and -not $DoNotUpdateDARKNET) {
  & $GIT_EXE pull
}

$CMAKE_EXE = Get-Command cmake 2> $null | Select-Object -ExpandProperty Definition
if (-Not $CMAKE_EXE) {
  throw "Could not find CMake, please install it"
}
else {
  Write-Host "Using CMake from ${CMAKE_EXE}"
}

if (-Not $DoNotUseNinja) {
  $NINJA_EXE = Get-Command ninja 2> $null | Select-Object -ExpandProperty Definition
  if (-Not $NINJA_EXE) {
    $DoNotUseNinja = $true
    Write-Host "Could not find Ninja, using msbuild or make backends as a fallback" -ForegroundColor Yellow
  }
  else {
    Write-Host "Using Ninja from ${NINJA_EXE}"
    $generator = "Ninja"
  }
}

function getProgramFiles32bit() {
  $out = ${env:PROGRAMFILES(X86)}
  if ($null -eq $out) {
    $out = ${env:PROGRAMFILES}
  }

  if ($null -eq $out) {
    throw "Could not find [Program Files 32-bit]"
  }

  return $out
}

function getLatestVisualStudioWithDesktopWorkloadPath() {
  $programFiles = getProgramFiles32bit
  $vswhereExe = "$programFiles\Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path $vswhereExe) {
    $output = & $vswhereExe -products * -latest -requires Microsoft.VisualStudio.Workload.NativeDesktop -format xml
    [xml]$asXml = $output
    foreach ($instance in $asXml.instances.instance) {
      $installationPath = $instance.InstallationPath -replace "\\$" # Remove potential trailing backslash
    }
    if (!$installationPath) {
      Write-Host "Warning: no full Visual Studio setup has been found, extending search to include also partial installations" -ForegroundColor Yellow
      $output = & $vswhereExe -products * -latest -format xml
      [xml]$asXml = $output
      foreach ($instance in $asXml.instances.instance) {
        $installationPath = $instance.InstallationPath -replace "\\$" # Remove potential trailing backslash
      }
    }
    if (!$installationPath) {
      Throw "Could not locate any installation of Visual Studio"
    }
  }
  else {
    Throw "Could not locate vswhere at $vswhereExe"
  }
  return $installationPath
}


function getLatestVisualStudioWithDesktopWorkloadVersion() {
  $programFiles = getProgramFiles32bit
  $vswhereExe = "$programFiles\Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path $vswhereExe) {
    $output = & $vswhereExe -products * -latest -requires Microsoft.VisualStudio.Workload.NativeDesktop -format xml
    [xml]$asXml = $output
    foreach ($instance in $asXml.instances.instance) {
      $installationVersion = $instance.InstallationVersion
    }
    if (!$installationVersion) {
      Write-Host "Warning: no full Visual Studio setup has been found, extending search to include also partial installations" -ForegroundColor Yellow
      $output = & $vswhereExe -products * -latest -format xml
      [xml]$asXml = $output
      foreach ($instance in $asXml.instances.instance) {
        $installationVersion = $instance.installationVersion
      }
    }
    if (!$installationVersion) {
      Throw "Could not locate any installation of Visual Studio"
    }
  }
  else {
    Throw "Could not locate vswhere at $vswhereExe"
  }
  return $installationVersion
}


if ((Test-Path env:VCPKG_ROOT) -and $UseVCPKG) {
  $vcpkg_path = "$env:VCPKG_ROOT"
  Write-Host "Found vcpkg in VCPKG_ROOT: $vcpkg_path"
  $additional_build_setup = $additional_build_setup + " -DENABLE_VCPKG_INTEGRATION:BOOL=ON"
}
elseif ((Test-Path "${env:WORKSPACE}/vcpkg") -and $UseVCPKG) {
  $vcpkg_path = "${env:WORKSPACE}/vcpkg"
  $env:VCPKG_ROOT = "${env:WORKSPACE}/vcpkg"
  Write-Host "Found vcpkg in WORKSPACE/vcpkg: $vcpkg_path"
  $additional_build_setup = $additional_build_setup + " -DENABLE_VCPKG_INTEGRATION:BOOL=ON"
}
elseif ((Test-Path "${RUNVCPKG_VCPKG_ROOT_OUT}") -and $UseVCPKG) {
  $vcpkg_path = "${RUNVCPKG_VCPKG_ROOT_OUT}"
  $env:VCPKG_ROOT = "${RUNVCPKG_VCPKG_ROOT_OUT}"
  Write-Host "Found vcpkg in RUNVCPKG_VCPKG_ROOT_OUT: ${RUNVCPKG_VCPKG_ROOT_OUT}"
  $additional_build_setup = $additional_build_setup + " -DENABLE_VCPKG_INTEGRATION:BOOL=ON"
}
elseif ($UseVCPKG) {
  if (-Not (Test-Path "$PWD/vcpkg")) {
    & $GIT_EXE clone https://github.com/microsoft/vcpkg
  }
  $vcpkg_path = "$PWD/vcpkg"
  $env:VCPKG_ROOT = "$PWD/vcpkg"
  Write-Host "Found vcpkg in $PWD/vcpkg: $PWD/vcpkg"
  $additional_build_setup = $additional_build_setup + " -DENABLE_VCPKG_INTEGRATION:BOOL=ON"
}
else {
  Write-Host "Skipping vcpkg integration`n" -ForegroundColor Yellow
  $additional_build_setup = $additional_build_setup + " -DENABLE_VCPKG_INTEGRATION:BOOL=OFF"
}

if ($UseVCPKG -and (Test-Path "$vcpkg_path/.git") -and -not $DoNotUpdateVCPKG) {
  Push-Location $vcpkg_path
  & $GIT_EXE pull
  & $PWD/bootstrap-vcpkg${bootstrap_ext} -disableMetrics
  Pop-Location
}

if (-Not $DoNotSetupVS) {
  if ($null -eq (Get-Command "cl.exe" -ErrorAction SilentlyContinue)) {
    $vsfound = getLatestVisualStudioWithDesktopWorkloadPath
    Write-Host "Found VS in ${vsfound}"
    Push-Location "${vsfound}\Common7\Tools"
    cmd.exe /c "VsDevCmd.bat -arch=x64 & set" |
    ForEach-Object {
      if ($_ -match "=") {
        $v = $_.split("="); Set-Item -force -path "ENV:\$($v[0])"  -value "$($v[1])"
      }
    }
    Pop-Location
    Write-Host "Visual Studio Command Prompt variables set"
  }

  $tokens = getLatestVisualStudioWithDesktopWorkloadVersion
  $tokens = $tokens.split('.')
  if ($DoNotUseNinja) {
    $dllfolder = "Release"
    $selectConfig = " --config Release "
    if ($tokens[0] -eq "14") {
      $generator = "Visual Studio 14 2015"
      $additional_build_setup = $additional_build_setup + " -T `"host=x64`" -A `"x64`""
    }
    elseif ($tokens[0] -eq "15") {
      $generator = "Visual Studio 15 2017"
      $additional_build_setup = $additional_build_setup + " -T `"host=x64`" -A `"x64`""
    }
    elseif ($tokens[0] -eq "16") {
      $generator = "Visual Studio 16 2019"
      $additional_build_setup = $additional_build_setup + " -T `"host=x64`" -A `"x64`""
    }
    else {
      throw "Unknown Visual Studio version, unsupported configuration"
    }
  }
  if (-Not $UseVCPKG) {
    $dllfolder = "../3rdparty/pthreads/bin"
  }
}
if ($DoNotSetupVS -and $DoNotUseNinja) {
  $generator = "Unix Makefiles"
}
Write-Host "Setting up environment to use CMake generator: $generator"

if (-Not $IsMacOS -and $EnableCUDA) {
  if ($null -eq (Get-Command "nvcc" -ErrorAction SilentlyContinue)) {
    if (Test-Path env:CUDA_PATH) {
      $env:PATH += ";${env:CUDA_PATH}/bin"
      Write-Host "Found cuda in ${env:CUDA_PATH}"
    }
    else {
      Write-Host "Unable to find CUDA, if necessary please install it or define a CUDA_PATH env variable pointing to the install folder" -ForegroundColor Yellow
    }
  }

  if (Test-Path env:CUDA_PATH) {
    if (-Not(Test-Path env:CUDA_TOOLKIT_ROOT_DIR)) {
      $env:CUDA_TOOLKIT_ROOT_DIR = "${env:CUDA_PATH}"
      Write-Host "Added missing env variable CUDA_TOOLKIT_ROOT_DIR" -ForegroundColor Yellow
    }
    if (-Not(Test-Path env:CUDACXX)) {
      $env:CUDACXX = "${env:CUDA_PATH}/bin/nvcc"
      Write-Host "Added missing env variable CUDACXX" -ForegroundColor Yellow
    }
  }
}

if ($ForceCPP) {
  $additional_build_setup = $additional_build_setup + " -DBUILD_AS_CPP:BOOL=ON"
}

if (-Not($EnableCUDA)) {
  $additional_build_setup = $additional_build_setup + " -DENABLE_CUDA:BOOL=OFF"
}

if (-Not($EnableCUDNN)) {
  $additional_build_setup = $additional_build_setup + " -DENABLE_CUDNN:BOOL=OFF"
}

if (-Not($EnableOPENCV)) {
  $additional_build_setup = $additional_build_setup + " -DENABLE_OPENCV:BOOL=OFF"
}

if (-Not($EnableOPENCV_CUDA)) {
  $additional_build_setup = $additional_build_setup + " -DVCPKG_BUILD_OPENCV_WITH_CUDA:BOOL=OFF"
}

$build_folder = "./build_release"
if (-Not $DoNotDeleteBuildFolder) {
  Write-Host "Removing folder $build_folder" -ForegroundColor Yellow
  Remove-Item -Force -Recurse -ErrorAction SilentlyContinue $build_folder
}

New-Item -Path $build_folder -ItemType directory -Force
Set-Location $build_folder
$cmake_args = "-G `"$generator`" ${additional_build_setup} -S .."
Write-Host "CMake args: $cmake_args"
$proc = Start-Process -NoNewWindow -PassThru -FilePath $CMAKE_EXE -ArgumentList $cmake_args
$proc.WaitForExit()
$exitCode = $proc.ExitCode
if (-not $exitCode -eq 0) {
  Throw "Config failed! Exited with $exitCode."
}
$proc = Start-Process -NoNewWindow -PassThru -FilePath $CMAKE_EXE -ArgumentList "--build . ${selectConfig} --parallel ${number_of_build_workers} --target install"
$proc.WaitForExit()
$exitCode = $proc.ExitCode
if (-not $exitCode -eq 0) {
  Throw "Config failed! Exited with $exitCode."
}
Remove-Item DarknetConfig.cmake
Remove-Item DarknetConfigVersion.cmake
$dllfiles = Get-ChildItem ./${dllfolder}/*.dll
if ($dllfiles) {
  Copy-Item $dllfiles ..
}
Set-Location ..
Copy-Item cmake/Modules/*.cmake share/darknet/
Pop-Location
