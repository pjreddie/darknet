#!/usr/bin/env pwsh

<#

.SYNOPSIS
        build
        Created By: Stefano Sinigardi
        Created Date: February 18, 2019
        Last Modified Date: November 10, 2021

.DESCRIPTION
Build darknet using CMake, trying to properly setup the environment around compiler

.PARAMETER DisableInteractive
Disable script interactivity (useful for CI runs)

.PARAMETER EnableCUDA
Enable CUDA feature

.PARAMETER EnableCUDNN
Enable CUDNN feature

.PARAMETER EnableOPENCV
Build darknet linking to OpenCV

.PARAMETER EnableOPENCV_CUDA
Use a CUDA-enabled OpenCV build

.PARAMETER UseVCPKG
Use VCPKG to build darknet dependencies. Clone it if not already found on system

.PARAMETER InstallDARKNETthroughVCPKG
Use VCPKG to install darknet thanks to the port integrated in it

.PARAMETER InstallDARKNETdependenciesThroughVCPKGManifest
Use VCPKG to install darknet dependencies using vcpkg manifest feature

.PARAMETER ForceVCPKGDarknetHEAD
Install darknet from vcpkg and force it to HEAD version, not latest port release

.PARAMETER DoNotUpdateVCPKG
Do not update vcpkg before running the build (valid only if vcpkg is cloned by this script or the version found on the system is git-enabled)

.PARAMETER DoNotUpdateDARKNET
Do not update darknet before running the build (valid only if darknet is git-enabled)

.PARAMETER DoNotDeleteBuildFolder
Do not delete temporary cmake build folder at the end of the script

.PARAMETER DoNotSetupVS
Do not setup VisualStudio environment using the vcvars script

.PARAMETER DoNotUseNinja
Do not use Ninja for build

.PARAMETER ForceCPP
Force building darknet using C++ compiler also for plain C code

.PARAMETER ForceStaticLib
Create darknet library as static instead of the default linking mode of your system

.PARAMETER ForceVCPKGCacheRemoval
Force clean up of the local vcpkg binary cache before building

.PARAMETER DoNotDeleteBuildtreesFolder
Do not delete vcpkg buildtrees temp folder at the end of the script

.PARAMETER ForceSetupVS
Forces Visual Studio setup, also on systems on which it would not have been enabled automatically

.PARAMETER EnableCSharpWrapper
Enables building C# darknet wrapper

.PARAMETER DownloadWeights
Download pre-trained weight files

.PARAMETER ForceGCCVersion
Force a specific GCC version

.PARAMETER ForceOpenCVVersion
Force a specific OpenCV version (valid only with vcpkg-enabled builds)

.PARAMETER NumberOfBuildWorkers
Forces a specific number of threads for parallel building

.PARAMETER AdditionalBuildSetup
Additional setup parameters to manually pass to CMake

.EXAMPLE
.\build -DisableInteractive -DoNotDeleteBuildFolder -UseVCPKG

#>

param (
  [switch]$DisableInteractive = $false,
  [switch]$EnableCUDA = $false,
  [switch]$EnableCUDNN = $false,
  [switch]$EnableOPENCV = $false,
  [switch]$EnableOPENCV_CUDA = $false,
  [switch]$UseVCPKG = $false,
  [switch]$InstallDARKNETthroughVCPKG = $false,
  [switch]$InstallDARKNETdependenciesThroughVCPKGManifest = $false,
  [switch]$ForceVCPKGDarknetHEAD = $false,
  [switch]$DoNotUpdateVCPKG = $false,
  [switch]$DoNotUpdateDARKNET = $false,
  [switch]$DoNotDeleteBuildFolder = $false,
  [switch]$DoNotSetupVS = $false,
  [switch]$DoNotUseNinja = $false,
  [switch]$ForceCPP = $false,
  [switch]$ForceStaticLib = $false,
  [switch]$ForceVCPKGCacheRemoval = $false,
  [switch]$DoNotDeleteBuildtreesFolder = $false,
  [switch]$ForceSetupVS = $false,
  [switch]$EnableCSharpWrapper = $false,
  [switch]$DownloadWeights = $false,
  [Int32]$ForceGCCVersion = 0,
  [Int32]$ForceOpenCVVersion = 0,
  [Int32]$NumberOfBuildWorkers = 8,
  [string]$AdditionalBuildSetup = ""  # "-DCMAKE_CUDA_ARCHITECTURES=30"
)

$build_ps1_version = "0.9.8"

$ErrorActionPreference = "SilentlyContinue"
Stop-Transcript | out-null
$ErrorActionPreference = "Continue"
Start-Transcript -Path $PSScriptRoot/build.log

Function MyThrow ($Message) {
  if ($DisableInteractive) {
    Write-Host $Message -ForegroundColor Red
    throw
  }
  else {
    # Check if running in PowerShell ISE
    if ($psISE) {
      # "ReadKey" not supported in PowerShell ISE.
      # Show MessageBox UI
      $Shell = New-Object -ComObject "WScript.Shell"
      $Shell.Popup($Message, 0, "OK", 0)
      throw
    }

    $Ignore =
    16, # Shift (left or right)
    17, # Ctrl (left or right)
    18, # Alt (left or right)
    20, # Caps lock
    91, # Windows key (left)
    92, # Windows key (right)
    93, # Menu key
    144, # Num lock
    145, # Scroll lock
    166, # Back
    167, # Forward
    168, # Refresh
    169, # Stop
    170, # Search
    171, # Favorites
    172, # Start/Home
    173, # Mute
    174, # Volume Down
    175, # Volume Up
    176, # Next Track
    177, # Previous Track
    178, # Stop Media
    179, # Play
    180, # Mail
    181, # Select Media
    182, # Application 1
    183  # Application 2

    Write-Host $Message -ForegroundColor Red
    Write-Host -NoNewline "Press any key to continue..."
    while (($null -eq $KeyInfo.VirtualKeyCode) -or ($Ignore -contains $KeyInfo.VirtualKeyCode)) {
      $KeyInfo = $Host.UI.RawUI.ReadKey("NoEcho, IncludeKeyDown")
    }
    Write-Host ""
    throw
  }
}

Function DownloadNinja() {
  Write-Host "Unable to find Ninja, downloading a portable version on-the-fly" -ForegroundColor Yellow
  Remove-Item -Force -Recurse -ErrorAction SilentlyContinue ninja
  Remove-Item -Force -ErrorAction SilentlyContinue ninja.zip
  if ($IsWindows -or $IsWindowsPowerShell) {
    $url = "https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-win.zip"
  }
  elseif ($IsLinux) {
    $url = "https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip"
  }
  elseif ($IsMacOS) {
    $url = "https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-mac.zip"
  }
  else {
    MyThrow("Unknown OS, unsupported")
  }
  Invoke-RestMethod -Uri $url -Method Get -ContentType application/zip -OutFile "ninja.zip"
  Expand-Archive -Path ninja.zip
  Remove-Item -Force -ErrorAction SilentlyContinue ninja.zip
}


Write-Host "Darknet build script version ${build_ps1_version}"

if ((-Not $DisableInteractive) -and (-Not $UseVCPKG)) {
  $Result = Read-Host "Enable vcpkg to install darknet dependencies (yes/no)"
  if (($Result -eq 'Yes') -or ($Result -eq 'Y') -or ($Result -eq 'yes') -or ($Result -eq 'y')) {
    $UseVCPKG = $true
  }
}

if ((-Not $DisableInteractive) -and (-Not $EnableCUDA) -and (-Not $IsMacOS)) {
  $Result = Read-Host "Enable CUDA integration (yes/no)"
  if (($Result -eq 'Yes') -or ($Result -eq 'Y') -or ($Result -eq 'yes') -or ($Result -eq 'y')) {
    $EnableCUDA = $true
  }
}

if ($EnableCUDA -and (-Not $DisableInteractive) -and (-Not $EnableCUDNN)) {
  $Result = Read-Host "Enable CUDNN optional dependency (yes/no)"
  if (($Result -eq 'Yes') -or ($Result -eq 'Y') -or ($Result -eq 'yes') -or ($Result -eq 'y')) {
    $EnableCUDNN = $true
  }
}

if ((-Not $DisableInteractive) -and (-Not $EnableOPENCV)) {
  $Result = Read-Host "Enable OpenCV optional dependency (yes/no)"
  if (($Result -eq 'Yes') -or ($Result -eq 'Y') -or ($Result -eq 'yes') -or ($Result -eq 'y')) {
    $EnableOPENCV = $true
  }
}

Write-Host -NoNewLine "PowerShell version:"
$PSVersionTable.PSVersion

if ($PSVersionTable.PSVersion.Major -eq 5) {
  $IsWindowsPowerShell = $true
}

if ($PSVersionTable.PSVersion.Major -lt 5) {
  MyThrow("Your PowerShell version is too old, please update it.")
}


if ($IsLinux -or $IsMacOS) {
  $bootstrap_ext = ".sh"
  $exe_ext = ""
}
elseif ($IsWindows -or $IsWindowsPowerShell) {
  $bootstrap_ext = ".bat"
  $exe_ext = ".exe"
}

if ($InstallDARKNETdependenciesThroughVCPKGManifest -and -not $InstallDARKNETthroughVCPKG) {
  Write-Host "You requested darknet dependencies to be installed by vcpkg in manifest mode but you didn't enable installation through vcpkg, doing that for you"
  $InstallDARKNETthroughVCPKG = $true
}

if ($InstallDARKNETthroughVCPKG -and -not $UseVCPKG) {
  Write-Host "You requested darknet to be installed by vcpkg but you didn't enable vcpkg, doing that for you"
  $UseVCPKG = $true
}

if ($InstallDARKNETthroughVCPKG -and -not $EnableOPENCV) {
  Write-Host "You requested darknet to be installed by vcpkg but you didn't enable OpenCV, doing that for you"
  $EnableOPENCV = $true
}

if ($UseVCPKG) {
  Write-Host "vcpkg bootstrap script: bootstrap-vcpkg${bootstrap_ext}"
}

if ((-Not $IsWindows) -and (-Not $IsWindowsPowerShell) -and (-Not $ForceSetupVS)) {
  $DoNotSetupVS = $true
}

if ($ForceStaticLib) {
  Write-Host "Forced CMake to produce a static library"
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DBUILD_SHARED_LIBS=OFF "
}

if (($IsLinux -or $IsMacOS) -and ($ForceGCCVersion -gt 0)) {
  Write-Host "Manually setting CC and CXX variables to gcc version $ForceGCCVersion"
  $env:CC = "gcc-$ForceGCCVersion"
  $env:CXX = "g++-$ForceGCCVersion"
}

if (($IsWindows -or $IsWindowsPowerShell) -and (-Not $env:VCPKG_DEFAULT_TRIPLET)) {
  $env:VCPKG_DEFAULT_TRIPLET = "x64-windows-release"
}
elseif ($IsMacOS -and (-Not $env:VCPKG_DEFAULT_TRIPLET)) {
  $env:VCPKG_DEFAULT_TRIPLET = "x64-osx-release"
}
elseif ($IsLinux -and (-Not $env:VCPKG_DEFAULT_TRIPLET)) {
  $env:VCPKG_DEFAULT_TRIPLET = "x64-linux-release"
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

if ($EnableCUDA -and $EnableOPENCV -and (-Not $EnableOPENCV_CUDA)) {
  Write-Host "OPENCV with CUDA extension is not enabled, you can enable it passing -EnableOPENCV_CUDA"
}
elseif ($EnableOPENCV -and $EnableOPENCV_CUDA -and (-Not $EnableCUDA)) {
  Write-Host "OPENCV with CUDA extension was requested, but CUDA is not enabled, you can enable it passing -EnableCUDA"
  $EnableOPENCV_CUDA = $false
}
elseif ($EnableCUDA -and $EnableOPENCV_CUDA -and (-Not $EnableOPENCV)) {
  Write-Host "OPENCV with CUDA extension was requested, but OPENCV is not enabled, you can enable it passing -EnableOPENCV"
  $EnableOPENCV_CUDA = $false
}
elseif ($EnableOPENCV_CUDA -and (-Not $EnableCUDA) -and (-Not $EnableOPENCV)) {
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

if ($EnableCSharpWrapper -and ($IsWindowsPowerShell -or $IsWindows)) {
  Write-Host "Yolo C# wrapper integration is enabled. Will be built with Visual Studio generator. Disabling Ninja"
  $DoNotUseNinja = $true
}
else {
  $EnableCSharpWrapper = $false
  Write-Host "Yolo C# wrapper integration is disabled, please pass -EnableCSharpWrapper to the script to enable. You must be on Windows!"
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

$GIT_EXE = Get-Command "git" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Definition
if (-Not $GIT_EXE) {
  MyThrow("Could not find git, please install it")
}
else {
  Write-Host "Using git from ${GIT_EXE}"
}

if (Test-Path "$PSScriptRoot/.git") {
  Write-Host "Darknet has been cloned with git and supports self-updating mechanism"
  if ($DoNotUpdateDARKNET) {
    Write-Host "Darknet will not self-update sources" -ForegroundColor Yellow
  }
  else {
    Write-Host "Darknet will self-update sources, please pass -DoNotUpdateDARKNET to the script to disable"
    $proc = Start-Process -NoNewWindow -PassThru -FilePath $GIT_EXE -ArgumentList "pull"
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Updating darknet sources failed! Exited with error code $exitCode.")
    }
  }
}

$CMAKE_EXE = Get-Command "cmake" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Definition
if (-Not $CMAKE_EXE) {
  MyThrow("Could not find CMake, please install it")
}
else {
  Write-Host "Using CMake from ${CMAKE_EXE}"
  $proc = Start-Process -NoNewWindow -PassThru -FilePath ${CMAKE_EXE} -ArgumentList "--version"
  $handle = $proc.Handle
  $proc.WaitForExit()
  $exitCode = $proc.ExitCode
  if (-Not ($exitCode -eq 0)) {
    MyThrow("CMake version check failed! Exited with error code $exitCode.")
  }
}

if (-Not $DoNotUseNinja) {
  $NINJA_EXE = Get-Command "ninja" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Definition
  if (-Not $NINJA_EXE) {
    DownloadNinja
    $env:PATH += ";${PSScriptRoot}/ninja"
    $NINJA_EXE = Get-Command "ninja" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Definition
    if (-Not $NINJA_EXE) {
      $DoNotUseNinja = $true
      Write-Host "Could not find Ninja, unable to download a portable ninja, using msbuild or make backends as a fallback" -ForegroundColor Yellow
    }
  }
  if ($NINJA_EXE) {
    Write-Host "Using Ninja from ${NINJA_EXE}"
    Write-Host -NoNewLine "Ninja version "
    $proc = Start-Process -NoNewWindow -PassThru -FilePath ${NINJA_EXE} -ArgumentList "--version"
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      $DoNotUseNinja = $true
      Write-Host "Unable to run Ninja previously found, using msbuild or make backends as a fallback" -ForegroundColor Yellow
    }
    else {
      $generator = "Ninja"
      $AdditionalBuildSetup = $AdditionalBuildSetup + " -DCMAKE_BUILD_TYPE=Release"
    }
  }
}

function getProgramFiles32bit() {
  $out = ${env:PROGRAMFILES(X86)}
  if ($null -eq $out) {
    $out = ${env:PROGRAMFILES}
  }

  if ($null -eq $out) {
    MyThrow("Could not find [Program Files 32-bit]")
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
      Write-Host "Warning: no full Visual Studio setup has been found, extending search to include also pre-release installations" -ForegroundColor Yellow
      $output = & $vswhereExe -prerelease -products * -latest -format xml
      [xml]$asXml = $output
      foreach ($instance in $asXml.instances.instance) {
        $installationPath = $instance.InstallationPath -replace "\\$" # Remove potential trailing backslash
      }
    }
    if (!$installationPath) {
      MyThrow("Could not locate any installation of Visual Studio")
    }
  }
  else {
    MyThrow("Could not locate vswhere at $vswhereExe")
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
      Write-Host "Warning: no full Visual Studio setup has been found, extending search to include also pre-release installations" -ForegroundColor Yellow
      $output = & $vswhereExe -prerelease -products * -latest -format xml
      [xml]$asXml = $output
      foreach ($instance in $asXml.instances.instance) {
        $installationVersion = $instance.installationVersion
      }
    }
    if (!$installationVersion) {
      MyThrow("Could not locate any installation of Visual Studio")
    }
  }
  else {
    MyThrow("Could not locate vswhere at $vswhereExe")
  }
  return $installationVersion
}

$vcpkg_root_set_by_this_script = $false

if ((Test-Path env:VCPKG_ROOT) -and $UseVCPKG) {
  $vcpkg_path = "$env:VCPKG_ROOT"
  Write-Host "Found vcpkg in VCPKG_ROOT: $vcpkg_path"
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_VCPKG_INTEGRATION:BOOL=ON"
}
elseif ((Test-Path "${env:WORKSPACE}/vcpkg") -and $UseVCPKG) {
  $vcpkg_path = "${env:WORKSPACE}/vcpkg"
  $env:VCPKG_ROOT = "${env:WORKSPACE}/vcpkg"
  $vcpkg_root_set_by_this_script = $true
  Write-Host "Found vcpkg in WORKSPACE/vcpkg: $vcpkg_path"
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_VCPKG_INTEGRATION:BOOL=ON"
}
elseif (-not($null -eq ${RUNVCPKG_VCPKG_ROOT_OUT})) {
  if ((Test-Path "${RUNVCPKG_VCPKG_ROOT_OUT}") -and $UseVCPKG) {
    $vcpkg_path = "${RUNVCPKG_VCPKG_ROOT_OUT}"
    $env:VCPKG_ROOT = "${RUNVCPKG_VCPKG_ROOT_OUT}"
    $vcpkg_root_set_by_this_script = $true
    Write-Host "Found vcpkg in RUNVCPKG_VCPKG_ROOT_OUT: ${vcpkg_path}"
    $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_VCPKG_INTEGRATION:BOOL=ON"
  }
}
elseif ($UseVCPKG) {
  if (-Not (Test-Path "$PWD/vcpkg")) {
    $proc = Start-Process -NoNewWindow -PassThru -FilePath $GIT_EXE -ArgumentList "clone https://github.com/microsoft/vcpkg"
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-not ($exitCode -eq 0)) {
      MyThrow("Cloning vcpkg sources failed! Exited with error code $exitCode.")
    }
  }
  $vcpkg_path = "$PWD/vcpkg"
  $env:VCPKG_ROOT = "$PWD/vcpkg"
  $vcpkg_root_set_by_this_script = $true
  Write-Host "Found vcpkg in $PWD/vcpkg: $PWD/vcpkg"
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_VCPKG_INTEGRATION:BOOL=ON"
}
else {
  Write-Host "Skipping vcpkg integration`n" -ForegroundColor Yellow
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_VCPKG_INTEGRATION:BOOL=OFF"
}

if ($UseVCPKG -and (Test-Path "$vcpkg_path/.git") -and (-Not $DoNotUpdateVCPKG)) {
  Push-Location $vcpkg_path
  $proc = Start-Process -NoNewWindow -PassThru -FilePath $GIT_EXE -ArgumentList "pull"
  $handle = $proc.Handle
  $proc.WaitForExit()
  $exitCode = $proc.ExitCode
  if (-Not ($exitCode -eq 0)) {
    MyThrow("Updating vcpkg sources failed! Exited with error code $exitCode.")
  }
  $proc = Start-Process -NoNewWindow -PassThru -FilePath $PWD/bootstrap-vcpkg${bootstrap_ext} -ArgumentList "-disableMetrics"
  $handle = $proc.Handle
  $proc.WaitForExit()
  $exitCode = $proc.ExitCode
  if (-Not ($exitCode -eq 0)) {
    MyThrow("Bootstrapping vcpkg failed! Exited with error code $exitCode.")
  }
  Pop-Location
}

if ($UseVCPKG -and ($vcpkg_path.length -gt 40) -and ($IsWindows -or $IsWindowsPowerShell)) {
  Write-Host "vcpkg path is very long and might fail. Please move it or" -ForegroundColor Yellow
  Write-Host "the entire darknet folder to a shorter path, like C:\darknet" -ForegroundColor Yellow
  Write-Host "You can use the subst command to ease the process if necessary" -ForegroundColor Yellow
  if (-Not $DisableInteractive) {
    $Result = Read-Host "Do you still want to continue? (yes/no)"
    if (($Result -eq 'No') -or ($Result -eq 'N') -or ($Result -eq 'no') -or ($Result -eq 'n')) {
      MyThrow("Build aborted")
    }
  }
}

if ($ForceVCPKGCacheRemoval -and (-Not $UseVCPKG)) {
  Write-Host "VCPKG is not enabled, so local vcpkg binary cache will not be deleted even if requested" -ForegroundColor Yellow
}

if ($UseVCPKG -and (-Not $DoNotDeleteBuildtreesFolder)) {
  Write-Host "Cleaning folder buildtrees inside vcpkg" -ForegroundColor Yellow
  Remove-Item -Force -Recurse -ErrorAction SilentlyContinue "$env:VCPKG_ROOT/buildtrees"
}

if (($ForceOpenCVVersion -eq 2) -and $UseVCPKG) {
  Write-Host "You requested OpenCV version 2, so vcpkg will install that version" -ForegroundColor Yellow
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DVCPKG_USE_OPENCV4=OFF -DVCPKG_USE_OPENCV2=ON"
}

if (($ForceOpenCVVersion -eq 3) -and $UseVCPKG) {
  Write-Host "You requested OpenCV version 3, so vcpkg will install that version" -ForegroundColor Yellow
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DVCPKG_USE_OPENCV4=OFF -DVCPKG_USE_OPENCV3=ON"
}

if ($UseVCPKG -and $ForceVCPKGCacheRemoval) {
  if ($IsWindows -or $IsWindowsPowerShell) {
    $vcpkgbinarycachepath = "$env:LOCALAPPDATA/vcpkg/archive"
  }
  elseif ($IsLinux) {
    $vcpkgbinarycachepath = "$env:HOME/.cache/vcpkg/archive"
  }
  elseif ($IsMacOS) {
    $vcpkgbinarycachepath = "$env:HOME/.cache/vcpkg/archive"
  }
  else {
    MyThrow("Unknown OS, unsupported")
  }
  Write-Host "Removing local vcpkg binary cache from $vcpkgbinarycachepath" -ForegroundColor Yellow
  Remove-Item -Force -Recurse -ErrorAction SilentlyContinue $vcpkgbinarycachepath
}

if (-Not $DoNotSetupVS) {
  $CL_EXE = Get-Command "cl" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Definition
  if ((-Not $CL_EXE) -or ($CL_EXE -match "HostX86\\x86") -or ($CL_EXE -match "HostX64\\x86")) {
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
      $AdditionalBuildSetup = $AdditionalBuildSetup + " -T `"host=x64`" -A `"x64`""
    }
    elseif ($tokens[0] -eq "15") {
      $generator = "Visual Studio 15 2017"
      $AdditionalBuildSetup = $AdditionalBuildSetup + " -T `"host=x64`" -A `"x64`""
    }
    elseif ($tokens[0] -eq "16") {
      $generator = "Visual Studio 16 2019"
      $AdditionalBuildSetup = $AdditionalBuildSetup + " -T `"host=x64`" -A `"x64`""
    }
    elseif ($tokens[0] -eq "17") {
      $generator = "Visual Studio 17 2022"
      $AdditionalBuildSetup = $AdditionalBuildSetup + " -T `"host=x64`" -A `"x64`""
    }
    else {
      MyThrow("Unknown Visual Studio version, unsupported configuration")
    }
  }
  if (-Not $UseVCPKG) {
    $dllfolder = "../3rdparty/pthreads/bin"
  }
}
if ($DoNotSetupVS -and $DoNotUseNinja) {
  $generator = "Unix Makefiles"
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DCMAKE_BUILD_TYPE=Release"
}
Write-Host "Setting up environment to use CMake generator: $generator"

if (-Not $IsMacOS -and $EnableCUDA) {
  $NVCC_EXE = Get-Command "nvcc" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Definition
  if (-Not $NVCC_EXE) {
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
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DBUILD_AS_CPP:BOOL=ON"
}

if (-Not $EnableCUDA) {
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_CUDA:BOOL=OFF"
}

if (-Not $EnableCUDNN) {
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_CUDNN:BOOL=OFF"
}

if (-Not $EnableOPENCV) {
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_OPENCV:BOOL=OFF"
}

if (-Not $EnableOPENCV_CUDA) {
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DVCPKG_BUILD_OPENCV_WITH_CUDA:BOOL=OFF"
}

if ($EnableCSharpWrapper) {
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_CSHARP_WRAPPER:BOOL=ON"
}

if ($InstallDARKNETthroughVCPKG) {
  if ($ForceVCPKGDarknetHEAD) {
    $headMode = " --head "
  }
  $features = "opencv-base"
  $feature_manifest_opencv = "--x-feature=opencv-base"
  if ($EnableCUDA) {
    $features = $features + ",cuda"
    $feature_manifest_cuda = "--x-feature=cuda"
  }
  if ($EnableCUDNN) {
    $features = $features + ",cudnn"
    $feature_manifest_cudnn = "--x-feature=cudnn"
  }
  if (-not (Test-Path "${env:VCPKG_ROOT}/vcpkg${exe_ext}")) {
    $proc = Start-Process -NoNewWindow -PassThru -FilePath ${env:VCPKG_ROOT}/bootstrap-vcpkg${bootstrap_ext} -ArgumentList "-disableMetrics"
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Bootstrapping vcpkg failed! Exited with error code $exitCode.")
    }
  }
  if ($InstallDARKNETdependenciesThroughVCPKGManifest) {
    Write-Host "Running vcpkg in manifest mode to install darknet dependencies"
    Write-Host "vcpkg install --x-no-default-features $feature_manifest_opencv $feature_manifest_cuda $feature_manifest_cudnn $headMode"
    $proc = Start-Process -NoNewWindow -PassThru -FilePath "${env:VCPKG_ROOT}/vcpkg${exe_ext}" -ArgumentList " install --x-no-default-features $feature_manifest_opencv $feature_manifest_cuda $feature_manifest_cudnn $headMode "
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Installing darknet through vcpkg failed! Exited with error code $exitCode.")
    }
  }
  else {
    Write-Host "Running vcpkg to install darknet"
    Write-Host "vcpkg install darknet[${features}] $headMode --recurse"
    Push-Location ${env:VCPKG_ROOT}
    if ($ForceVCPKGDarknetHEAD) {
      $proc = Start-Process -NoNewWindow -PassThru -FilePath "${env:VCPKG_ROOT}/vcpkg${exe_ext}" -ArgumentList " --feature-flags=-manifests remove darknet --recurse "
      $handle = $proc.Handle
      $proc.WaitForExit()
      $exitCode = $proc.ExitCode
      if (-Not ($exitCode -eq 0)) {
        MyThrow("Removing darknet through vcpkg failed! Exited with error code $exitCode.")
      }
    }
    $proc = Start-Process -NoNewWindow -PassThru -FilePath "${env:VCPKG_ROOT}/vcpkg${exe_ext}" -ArgumentList " --feature-flags=-manifests upgrade --no-dry-run "
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Upgrading vcpkg installed ports failed! Exited with error code $exitCode.")
    }
    $proc = Start-Process -NoNewWindow -PassThru -FilePath "${env:VCPKG_ROOT}/vcpkg${exe_ext}" -ArgumentList " --feature-flags=-manifests install darknet[${features}] $headMode --recurse "  # "-manifest"  disables the manifest feature, so that if vcpkg is a subfolder of darknet, the vcpkg.json inside darknet folder does not trigger errors due to automatic manifest mode
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Installing darknet dependencies through vcpkg failed! Exited with error code $exitCode.")
    }
    Pop-Location
  }
}
else {
  $build_folder = "./build_release"
  if (-Not $DoNotDeleteBuildFolder) {
    Write-Host "Removing folder $build_folder" -ForegroundColor Yellow
    Remove-Item -Force -Recurse -ErrorAction SilentlyContinue $build_folder
  }
  New-Item -Path $build_folder -ItemType directory -Force | Out-Null
  Set-Location $build_folder
  $cmake_args = "-G `"$generator`" ${AdditionalBuildSetup} -S .."
  Write-Host "Configuring CMake project" -ForegroundColor Green
  Write-Host "CMake args: $cmake_args"
  $proc = Start-Process -NoNewWindow -PassThru -FilePath $CMAKE_EXE -ArgumentList $cmake_args
  $handle = $proc.Handle
  $proc.WaitForExit()
  $exitCode = $proc.ExitCode
  if (-Not ($exitCode -eq 0)) {
    MyThrow("Config failed! Exited with error code $exitCode.")
  }
  Write-Host "Building CMake project" -ForegroundColor Green
  $proc = Start-Process -NoNewWindow -PassThru -FilePath $CMAKE_EXE -ArgumentList "--build . ${selectConfig} --parallel ${NumberOfBuildWorkers} --target install"
  $handle = $proc.Handle
  $proc.WaitForExit()
  $exitCode = $proc.ExitCode
  if (-Not ($exitCode -eq 0)) {
    MyThrow("Config failed! Exited with error code $exitCode.")
  }
  Remove-Item -Force -ErrorAction SilentlyContinue DarknetConfig.cmake
  Remove-Item -Force -ErrorAction SilentlyContinue DarknetConfigVersion.cmake
  $dllfiles = Get-ChildItem ./${dllfolder}/*.dll
  if ($dllfiles) {
    Copy-Item $dllfiles ..
  }
  Set-Location ..
  Copy-Item cmake/Modules/*.cmake share/darknet/
  Pop-Location
}

if ($UseVCPKG -and (-Not $DoNotDeleteBuildtreesFolder)) {
  Write-Host "Cleaning folder buildtrees inside vcpkg" -ForegroundColor Yellow
  Remove-Item -Force -Recurse -ErrorAction SilentlyContinue "$env:VCPKG_ROOT/buildtrees"
}

Write-Host "Build complete!" -ForegroundColor Green

if ($DownloadWeights) {
  Write-Host "Downloading weights..." -ForegroundColor Yellow
  & $PSScriptRoot/scripts/download_weights.ps1
  Write-Host "Weights downloaded" -ForegroundColor Green
}

if ($vcpkg_root_set_by_this_script) {
  $env:VCPKG_ROOT = $null
}

$ErrorActionPreference = "SilentlyContinue"
Stop-Transcript | out-null
$ErrorActionPreference = "Continue"
