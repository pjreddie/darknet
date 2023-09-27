#!/usr/bin/env pwsh

<#

.SYNOPSIS
        build
        Created By: Stefano Sinigardi
        Created Date: February 18, 2019
        Last Modified Date: September 25, 2023

.DESCRIPTION
Build tool using CMake, trying to properly setup the environment around compiler

.PARAMETER DisableInteractive
Disable script interactivity (useful for CI runs)

.PARAMETER BuildDebug
Build using the debug toolchain

.PARAMETER DisableDLLcopy
Disable automatic DLL deployment through vcpkg at the end

.PARAMETER EnableCUDA
Enable CUDA feature

.PARAMETER EnableCUDNN
Enable CUDNN feature

.PARAMETER EnableOPENCV
Enable OpenCV feature

.PARAMETER ForceOpenCVVersion
Force a specific OpenCV version

.PARAMETER EnableOPENCV_CUDA
Use a CUDA-enabled OpenCV build

.PARAMETER UseVCPKG
Use vcpkg to build tool dependencies. Clone it if not already found on system

.PARAMETER ForceLocalVCPKG
Use a copy of vcpkg in a subfolder of the tool folder, even if there might be another copy already provided by the system

.PARAMETER InstallDARKNETthroughVCPKG
Use VCPKG to install darknet thanks to the port integrated in it

.PARAMETER InstallDARKNETdependenciesThroughVCPKGManifest
Use VCPKG to install darknet dependencies using vcpkg manifest feature

.PARAMETER ForceVCPKGDarknetHEAD
Install darknet from vcpkg and force it to HEAD version, not latest port release

.PARAMETER DoNotUpdateVCPKG
Do not update vcpkg before running the build (valid only if vcpkg is cloned by this script or the version found on the system is git-enabled)

.PARAMETER VCPKGSuffix
Specify a suffix to the vcpkg local folder for searching, useful to point to a custom version

.PARAMETER VCPKGFork
Specify a fork username to point to a custom version of vcpkg (ex: -VCPKGFork "custom" to point to github.com/custom/vcpkg)

.PARAMETER VCPKGBranch
Specify a branch to checkout in the vcpkg folder, useful to point to a custom version especially for forked vcpkg versions

.PARAMETER DoNotUpdateTOOL
Do not update the tool before running the build (valid only if tool is git-enabled)

.PARAMETER DoNotDeleteBuildFolder
Do not delete temporary cmake build folder at the end of the script

.PARAMETER DoNotSetupVS
Do not setup VisualStudio environment using the vcvars script

.PARAMETER DoNotUseNinja
Do not use Ninja for build

.PARAMETER ForceCPP
Force building darknet using C++ compiler also for plain C code

.PARAMETER ForceStaticLib
Create library as static instead of the default linking mode of your system

.PARAMETER ForceVCPKGCacheRemoval
Force clean up of the local vcpkg binary cache before building

.PARAMETER ForceVCPKGBuildtreesRemoval
Force clean up of vcpkg buildtrees temp folder at the end of the script

.PARAMETER ForceVCPKGBuildtreesPath
Force using a different buildtrees dir for vcpkg

.PARAMETER ForceVCPKGPackagesRemoval
Force clean up of vcpkg packages folder at the end of the script

.PARAMETER ForceSetupVS
Forces Visual Studio setup, also on systems on which it would not have been enabled automatically

.PARAMETER ForceCMakeFromVS
Forces usage of CMake from Visual Studio instead of the system-wide/user installed one

.PARAMETER ForceNinjaFromVS
Forces usage of Ninja from Visual Studio instead of the system-wide/user installed one

.PARAMETER EnableCSharpWrapper
Enables building C# darknet wrapper

.PARAMETER DownloadWeights
Download pre-trained weight files

.PARAMETER Use32bitTriplet
Use 32 bit triplet for target build (windows-only)

.PARAMETER BuildInstaller
Build an installer using CPack

.PARAMETER ForceGCCVersion
Force a specific GCC version

.PARAMETER NumberOfBuildWorkers
Forces a specific number of threads for parallel building

.PARAMETER AdditionalBuildSetup
Additional setup parameters to manually pass to CMake

.EXAMPLE
./build -DisableInteractive -DoNotDeleteBuildFolder -UseVCPKG

#>

<#
Copyright (c) Stefano Sinigardi

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
#>

param (
  [switch]$DisableInteractive = $false,
  [switch]$BuildDebug = $false,
  [switch]$DisableDLLcopy = $false,
  [switch]$EnableCUDA = $false,
  [switch]$EnableCUDNN = $false,
  [switch]$EnableOPENCV = $false,
  [Int32]$ForceOpenCVVersion = 0,
  [switch]$EnableOPENCV_CUDA = $false,
  [switch]$UseVCPKG = $false,
  [switch]$ForceLocalVCPKG = $false,
  [switch]$InstallDARKNETthroughVCPKG = $false,
  [switch]$InstallDARKNETdependenciesThroughVCPKGManifest = $false,
  [switch]$ForceVCPKGDarknetHEAD = $false,
  [switch]$DoNotUpdateVCPKG = $false,
  [string]$VCPKGSuffix = "",
  [string]$VCPKGFork = "",
  [string]$VCPKGBranch = "",
  [switch]$DoNotUpdateTOOL = $false,
  [switch]$DoNotDeleteBuildFolder = $false,
  [switch]$DoNotSetupVS = $false,
  [switch]$DoNotUseNinja = $false,
  [switch]$ForceCPP = $false,
  [switch]$ForceStaticLib = $false,
  [switch]$ForceVCPKGCacheRemoval = $false,
  [switch]$ForceVCPKGBuildtreesRemoval = $false,
  [string]$ForceVCPKGBuildtreesPath = "",
  [switch]$ForceVCPKGPackagesRemoval = $false,
  [switch]$ForceSetupVS = $false,
  [switch]$ForceCMakeFromVS = $false,
  [switch]$ForceNinjaFromVS = $false,
  [switch]$EnableCSharpWrapper = $false,
  [switch]$DownloadWeights = $false,
  [switch]$Use32bitTriplet = $false,
  [switch]$BuildInstaller = $false,
  [Int32]$ForceGCCVersion = 0,
  [Int32]$NumberOfBuildWorkers = 8,
  [string]$AdditionalBuildSetup = ""  # "-DCMAKE_CUDA_ARCHITECTURES=30"
)

$global:DisableInteractive = $DisableInteractive

$build_ps1_version = "3.6.1"
$script_name = $MyInvocation.MyCommand.Name
$utils_psm1_avail = $false

if (Test-Path $PSScriptRoot/scripts/utils.psm1) {
  Import-Module -Name $PSScriptRoot/scripts/utils.psm1 -Force
  $utils_psm1_avail = $true
}
else {
  $utils_psm1_version = "unavail"
  $IsWindowsPowerShell = $false
  $IsInGitSubmodule = $false
}

if (-Not $utils_psm1_avail) {
  $DoNotSetupVS = $true
  $ForceCMakeFromVS = $false
}

$ErrorActionPreference = "SilentlyContinue"
Stop-Transcript | out-null
$ErrorActionPreference = "Continue"
if($IsInGitSubmodule) {
  $PSCustomScriptRoot = Split-Path $PSScriptRoot -Parent
}
else {
  $PSCustomScriptRoot = $PSScriptRoot
}
$BuildLogPath = "$PSCustomScriptRoot/build.log"
$ReleaseInstallPrefix = "$PSCustomScriptRoot"
$DebugInstallPrefix = "$PSCustomScriptRoot/debug"
$DebugBuildSetup = " -DCMAKE_BUILD_TYPE=Debug "
$ReleaseBuildSetup = " -DCMAKE_BUILD_TYPE=Release "
if (-Not $BuildInstaller) {
  $DebugBuildSetup = $DebugBuildSetup + " -DCMAKE_INSTALL_PREFIX=$DebugInstallPrefix "
  $ReleaseBuildSetup = $ReleaseBuildSetup + " -DCMAKE_INSTALL_PREFIX=$ReleaseInstallPrefix "
}
Start-Transcript -Path $BuildLogPath

Write-Host "Build script version ${build_ps1_version}, utils module version ${utils_psm1_version}"
if (-Not $utils_psm1_avail) {
  Write-Host "utils.psm1 is not available, so VS integration is forcefully disabled" -ForegroundColor Yellow
}
Write-Host "Working directory: $PSCustomScriptRoot, log file: $BuildLogPath, $script_name is in submodule: $IsInGitSubmodule"

if ((-Not $global:DisableInteractive) -and (-Not $UseVCPKG)) {
  $Result = Read-Host "Enable vcpkg to install dependencies (yes/no)"
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

if ($IsWindowsPowerShell) {
  Write-Host "Running on Windows Powershell, please consider update and running on newer Powershell versions"
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
if ($ForceLocalVCPKG -And -Not $UseVCPKG) {
  $UseVCPKG = $true
  Write-Host "ForceLocalVCPKG was true but UseVCPKG was false, setting UseVCPKG to true"
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

$vcpkg_triplet_set_by_this_script = $false
$vcpkg_host_triplet_set_by_this_script = $false

if (($IsWindows -or $IsWindowsPowerShell) -and (-Not $env:VCPKG_DEFAULT_TRIPLET)) {
  if ($Use32bitTriplet) {
    if (-Not $BuildDebug) {
      $BuildDebug = $true
      Write-Host "Warning: when building for 32bit windows target, only builds with also debug version are possible. Debug has been enabled on your behalf!" -ForegroundColor Yellow
    }
    if (-Not $DoNotUseNinja) {
      $DoNotUseNinja = $true
      Write-Host "Warning: when building for 32bit windows target, only msbuild can be used and ninja will be disabled. Doing that for you!" -ForegroundColor Yellow
    }
    $env:VCPKG_DEFAULT_TRIPLET = "x86-windows"
    $vcpkg_triplet_set_by_this_script = $true
  }
  else {
    if($BuildDebug) {
      $env:VCPKG_DEFAULT_TRIPLET = "x64-windows"
      $vcpkg_triplet_set_by_this_script = $true
    }
    else {
      $env:VCPKG_DEFAULT_TRIPLET = "x64-windows-release"
      $vcpkg_triplet_set_by_this_script = $true
    }
  }
}
if (($IsWindows -or $IsWindowsPowerShell) -and (-Not $env:VCPKG_DEFAULT_HOST_TRIPLET)) {
  if ($BuildDebug) {
    $env:VCPKG_DEFAULT_HOST_TRIPLET = "x64-windows"
    $vcpkg_host_triplet_set_by_this_script = $true
  }
  else {
    $env:VCPKG_DEFAULT_HOST_TRIPLET = "x64-windows-release"
    $vcpkg_host_triplet_set_by_this_script = $true
  }
}

if ($IsMacOS -and (-Not $env:VCPKG_DEFAULT_TRIPLET)) {
  if ($BuildDebug) {
    $env:VCPKG_DEFAULT_TRIPLET = "x64-osx"
    $vcpkg_triplet_set_by_this_script = $true
  }
  else {
    $env:VCPKG_DEFAULT_TRIPLET = "x64-osx-release"
    $vcpkg_triplet_set_by_this_script = $true
  }
}
if ($IsMacOS -and (-Not $env:VCPKG_DEFAULT_HOST_TRIPLET)) {
  if ($BuildDebug) {
    $env:VCPKG_DEFAULT_HOST_TRIPLET = "x64-osx"
    $vcpkg_host_triplet_set_by_this_script = $true
  }
  else {
    $env:VCPKG_DEFAULT_HOST_TRIPLET = "x64-osx-release"
    $vcpkg_host_triplet_set_by_this_script = $true
  }
}

if ($IsLinux -and (-Not $env:VCPKG_DEFAULT_TRIPLET)) {
  if ($true) {
    if ($BuildDebug) {
      $env:VCPKG_DEFAULT_TRIPLET = "x64-linux"
      $vcpkg_triplet_set_by_this_script = $true
    }
    else {
      $env:VCPKG_DEFAULT_TRIPLET = "x64-linux-release"
      $vcpkg_triplet_set_by_this_script = $true
    }
  }
}
if ($IsLinux -and (-Not $env:VCPKG_DEFAULT_HOST_TRIPLET)) {
  if ($BuildDebug) {
    $env:VCPKG_DEFAULT_HOST_TRIPLET = "x64-linux"
    $vcpkg_host_triplet_set_by_this_script = $true
  }
  else {
    $env:VCPKG_DEFAULT_HOST_TRIPLET = "x64-linux-release"
    $vcpkg_host_triplet_set_by_this_script = $true
  }
}

if ($VCPKGSuffix -ne "" -and -not $UseVCPKG) {
  Write-Host "You specified a vcpkg folder suffix but didn't enable vcpkg integration, doing that for you" -ForegroundColor Yellow
  $UseVCPKG = $true
}

if ($VCPKGFork -ne "" -and -not $UseVCPKG) {
  Write-Host "You specified a vcpkg fork but didn't enable vcpkg integration, doing that for you" -ForegroundColor Yellow
  $UseVCPKG = $true
}

if ($VCPKGBranch -ne "" -and -not $UseVCPKG) {
  Write-Host "You specified a vcpkg branch but didn't enable vcpkg integration, doing that for you" -ForegroundColor Yellow
  $UseVCPKG = $true
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
  if ($BuildDebug -and ($env:VCPKG_DEFAULT_TRIPLET -match "release")) {
    MyThrow("You asked to build also debug binaries but VCPKG_DEFAULT_TRIPLET is set to a release triplet")
  }
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

Push-Location $PSCustomScriptRoot

$GIT_EXE = Get-Command "git" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Definition
if (-Not $GIT_EXE) {
  MyThrow("Could not find git, please install it")
}
else {
  Write-Host "Using git from ${GIT_EXE}"
}

$GitRepoPath = Resolve-Path "$PSCustomScriptRoot/.git" -ErrorAction SilentlyContinue
$GitModulesPath = Resolve-Path "$PSCustomScriptRoot/.gitmodules" -ErrorAction SilentlyContinue
if (Test-Path "$GitRepoPath") {
  Write-Host "This tool has been cloned with git and supports self-updating mechanism"
  if ($DoNotUpdateTOOL) {
    Write-Host "This tool will not self-update sources" -ForegroundColor Yellow
  }
  else {
    Write-Host "This tool will self-update sources, please pass -DoNotUpdateTOOL to the script to disable"
    Set-Location "$PSCustomScriptRoot"
    $proc = Start-Process -NoNewWindow -PassThru -FilePath $GIT_EXE -ArgumentList "pull"
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Updating this tool sources failed! Exited with error code $exitCode.")
    }
    if (Test-Path "$GitModulesPath") {
      $proc = Start-Process -NoNewWindow -PassThru -FilePath $GIT_EXE -ArgumentList "submodule update --init --recursive"
      $handle = $proc.Handle
      $proc.WaitForExit()
      $exitCode = $proc.ExitCode
      if (-Not ($exitCode -eq 0)) {
        MyThrow("Updating this tool submodule sources failed! Exited with error code $exitCode.")
      }
    }
    Set-Location "$PSCustomScriptRoot"
  }
}

if ($ForceCmakeFromVS) {
  $vsfound = getLatestVisualStudioWithDesktopWorkloadPath
  $cmakePath = "${vsfound}/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin"
  $vsCmakePath = "${vsfound}/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
  $CMAKE_EXE = Get-Command "cmake" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Definition
  if ((Test-Path "$vsCmakePath") -and -not ($vsCmakePath -eq $CMAKE_EXE)) {
    Write-Host "Adding CMake from Visual Studio to PATH"
    $env:PATH = '{0}{1}{2}' -f "$cmakePath", [IO.Path]::PathSeparator, $env:PATH
  }
  elseif ($vsCmakePath -eq $CMAKE_EXE) {
    Write-Host "CMake from Visual Studio was already the preferred choice" -ForegroundColor Yellow
  }
  else {
    Write-Host "Unable to find CMake integrated in Visual Studio" -ForegroundColor Red
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
  if ($ForceNinjaFromVS) {
    $vsfound = getLatestVisualStudioWithDesktopWorkloadPath
    $ninjaPath = "${vsfound}/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja"
    $vsninjaPath = "${vsfound}/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"
    $NINJA_EXE = Get-Command "ninja" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Definition
    if ((Test-Path "$vsninjaPath") -and -not ($vsninjaPath -eq $NINJA_EXE) -and (-not $DoNotUseNinja)) {
      Write-Host "Adding Ninja from Visual Studio to PATH"
      $env:PATH = '{0}{1}{2}' -f "$ninjaPath", [IO.Path]::PathSeparator, $env:PATH
    }
    elseif ($vsninjaPath -eq $NINJA_EXE) {
      Write-Host "Ninja from Visual Studio was already the preferred choice" -ForegroundColor Yellow
    }
    else {
      Write-Host "Unable to find Ninja integrated in Visual Studio" -ForegroundColor Red
    }
  }
  $NINJA_EXE = Get-Command "ninja" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Definition
  if (-Not $NINJA_EXE) {
    DownloadNinja
    $NinjaPath = Join-Path (${PSCustomScriptRoot}) 'ninja'
    $env:PATH = '{0}{1}{2}' -f $env:PATH, [IO.Path]::PathSeparator, "$NinjaPath"
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
    }
  }
}

if (-Not $DoNotSetupVS) {
  $CL_EXE = Get-Command "cl" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Definition
  if ((-Not $CL_EXE) -or ($CL_EXE -match "HostX86\\x86") -or ($CL_EXE -match "HostX64\\x86")) {
    $vsfound = getLatestVisualStudioWithDesktopWorkloadPath
    Write-Host "Found VS in ${vsfound}"
    Push-Location "${vsfound}/Common7/Tools"
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
    $debugConfig = " --config Debug "
    $releaseConfig = " --config Release "
    if ($Use32bitTriplet) {
      $targetArchitecture = "`"Win32`""
    }
    else {
      $targetArchitecture = "`"x64`""
    }
    if ($tokens[0] -eq "14") {
      $generator = "Visual Studio 14 2015"
      $AdditionalBuildSetup = $AdditionalBuildSetup + " -T `"host=x64`" -A $targetArchitecture"
    }
    elseif ($tokens[0] -eq "15") {
      $generator = "Visual Studio 15 2017"
      $AdditionalBuildSetup = $AdditionalBuildSetup + " -T `"host=x64`" -A $targetArchitecture"
    }
    elseif ($tokens[0] -eq "16") {
      $generator = "Visual Studio 16 2019"
      $AdditionalBuildSetup = $AdditionalBuildSetup + " -T `"host=x64`" -A $targetArchitecture"
    }
    elseif ($tokens[0] -eq "17") {
      $generator = "Visual Studio 17 2022"
      $AdditionalBuildSetup = $AdditionalBuildSetup + " -T `"host=x64`" -A $targetArchitecture"
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
}
Write-Host "Setting up environment to use CMake generator: $generator"

if (-Not $IsMacOS -and $EnableCUDA) {
  $NVCC_EXE = Get-Command "nvcc" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Definition
  if (-Not $NVCC_EXE) {
    if (Test-Path env:CUDA_PATH) {
      $env:PATH = '{0}{1}{2}' -f $env:PATH, [IO.Path]::PathSeparator, "${env:CUDA_PATH}/bin"
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

$vcpkg_root_set_by_this_script = $false

if ($UseVCPKG -And -Not $ForceLocalVCPKG) {
  if ((Test-Path env:VCPKG_ROOT) -and $VCPKGSuffix -eq "") {
    $vcpkg_path = "$env:VCPKG_ROOT"
    $vcpkg_path = Resolve-Path $vcpkg_path
    Write-Host "Found vcpkg in VCPKG_ROOT: $vcpkg_path"
    $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_VCPKG_INTEGRATION:BOOL=ON"
  }
  elseif (-not($null -eq ${env:WORKSPACE}) -and (Test-Path "${env:WORKSPACE}/vcpkg${VCPKGSuffix}")) {
    $vcpkg_path = "${env:WORKSPACE}/vcpkg${VCPKGSuffix}"
    $vcpkg_path = Resolve-Path $vcpkg_path
    $env:VCPKG_ROOT = "$vcpkg_path"
    $vcpkg_root_set_by_this_script = $true
    Write-Host "Found vcpkg in WORKSPACE/vcpkg${VCPKGSuffix}: $vcpkg_path"
    $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_VCPKG_INTEGRATION:BOOL=ON"
  }
  elseif (-not($null -eq ${RUNVCPKG_VCPKG_ROOT_OUT})) {
    if ((Test-Path "${RUNVCPKG_VCPKG_ROOT_OUT}") -and $UseVCPKG) {
      $vcpkg_path = "${RUNVCPKG_VCPKG_ROOT_OUT}"
      $vcpkg_path = Resolve-Path $vcpkg_path
      $env:VCPKG_ROOT = "$vcpkg_path"
      $vcpkg_root_set_by_this_script = $true
      Write-Host "Found vcpkg in RUNVCPKG_VCPKG_ROOT_OUT: $vcpkg_path"
      $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_VCPKG_INTEGRATION:BOOL=ON"
    }
  }
}
if (($null -eq $vcpkg_path) -and $UseVCPKG) {
  if (-Not (Test-Path "$PWD/vcpkg${VCPKGSuffix}")) {
    $shallow_copy = ""
    if(($ForceOpenCVVersion -eq 0)) {
      $shallow_copy = " --depth 1 "
    }
    $proc = Start-Process -NoNewWindow -PassThru -FilePath $GIT_EXE -ArgumentList "clone $shallow_copy https://github.com/microsoft/vcpkg vcpkg${VCPKGSuffix}"
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-not ($exitCode -eq 0)) {
      MyThrow("Cloning vcpkg sources failed! Exited with error code $exitCode.")
    }
  }
  $vcpkg_path = "$PWD/vcpkg${VCPKGSuffix}"
  $vcpkg_path = Resolve-Path $vcpkg_path
  $env:VCPKG_ROOT = "$vcpkg_path"
  $vcpkg_root_set_by_this_script = $true
  Write-Host "Found vcpkg in $PWD/vcpkg${VCPKGSuffix}: $vcpkg_path"
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_VCPKG_INTEGRATION:BOOL=ON"
}

$vcpkg_branch_set_by_this_script = $false

if ($UseVCPKG -and (Test-Path "$vcpkg_path/.git")) {
  Push-Location $vcpkg_path
  if ($VCPKGFork -ne "") {
    $vcpkgfork_already_setup = $false
    $remotes = & $GIT_EXE 'remote'
    ForEach ($remote in $remotes) {
      if ($remote -eq "vcpkgfork") {
        $vcpkgfork_already_setup = $true
        Write-Host "remote vcpkgfork already setup"
      }
    }
    if (-Not $vcpkgfork_already_setup) {
      $git_args = "remote add vcpkgfork https://github.com/${VCPKGFork}/vcpkg"
      Write-Host "setting up remote vcpkgfork"
      $proc = Start-Process -NoNewWindow -PassThru -FilePath $GIT_EXE -ArgumentList "$git_args"
      $handle = $proc.Handle
      $proc.WaitForExit()
      $exitCode = $proc.ExitCode
      if (-Not ($exitCode -eq 0)) {
        MyThrow("Adding remote https://github.com/${VCPKGFork}/vcpkg failed! Exited with error code $exitCode.")
      }
    }
    $git_args = "fetch vcpkgfork"
    $proc = Start-Process -NoNewWindow -PassThru -FilePath $GIT_EXE -ArgumentList "$git_args"
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Fetching from remote https://github.com/${VCPKGFork}/vcpkg failed! Exited with error code $exitCode.")
    }
  }
  if ($VCPKGBranch -ne "") {
    if ($VCPKGFork -ne "") {
      $git_args = "checkout vcpkgfork/$VCPKGBranch"
    }
    else {
      $git_args = "checkout $VCPKGBranch"
    }
    $proc = Start-Process -NoNewWindow -PassThru -FilePath $GIT_EXE -ArgumentList "$git_args"
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Checking out branch $VCPKGBranch failed! Exited with error code $exitCode.")
    }
    $vcpkg_branch_set_by_this_script = $true
  }
  if (-Not $DoNotUpdateVCPKG -and $VCPKGFork -eq "") {
    $proc = Start-Process -NoNewWindow -PassThru -FilePath $GIT_EXE -ArgumentList "pull"
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Updating vcpkg sources failed! Exited with error code $exitCode.")
    }
    $VcpkgBootstrapScript = Join-Path $PWD "bootstrap-vcpkg${bootstrap_ext}"
    $proc = Start-Process -NoNewWindow -PassThru -FilePath $VcpkgBootstrapScript -ArgumentList "-disableMetrics"
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Bootstrapping vcpkg failed! Exited with error code $exitCode.")
    }
  }
  Pop-Location
}

if ($UseVCPKG -and ($vcpkg_path.length -gt 40) -and ($IsWindows -or $IsWindowsPowerShell)) {
  Write-Host "vcpkg path is very long and might fail. Please move it or" -ForegroundColor Yellow
  Write-Host "the entire tool folder to a shorter path, like C:\src" -ForegroundColor Yellow
  Write-Host "You can use the subst command to ease the process if necessary" -ForegroundColor Yellow
  if (-Not $global:DisableInteractive) {
    $Result = Read-Host "Do you still want to continue? (yes/no)"
    if (($Result -eq 'No') -or ($Result -eq 'N') -or ($Result -eq 'no') -or ($Result -eq 'n')) {
      MyThrow("Build aborted")
    }
  }
}

if ($ForceVCPKGCacheRemoval -and (-Not $UseVCPKG)) {
  Write-Host "VCPKG is not enabled, so local vcpkg binary cache will not be deleted even if requested" -ForegroundColor Yellow
}

if ($BuildInstaller) {
  Write-Host "You requested to build an installer, so enabling this option if supported" -ForegroundColor Yellow
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_INSTALLER=ON"
}

if (($ForceOpenCVVersion -eq 2) -and $UseVCPKG) {
  Write-Host "You requested OpenCV version 2, so vcpkg will install that version" -ForegroundColor Yellow
  Write-Host "This requires using vcpkg.json.opencv23 as manifest file" -ForegroundColor Yellow
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DVCPKG_USE_OPENCV4=OFF -DVCPKG_USE_OPENCV2=ON"
}

if (($ForceOpenCVVersion -eq 3) -and $UseVCPKG) {
  Write-Host "You requested OpenCV version 3, so vcpkg will install that version" -ForegroundColor Yellow
  Write-Host "This requires using vcpkg.json.opencv23 as manifest file" -ForegroundColor Yellow
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DVCPKG_USE_OPENCV4=OFF -DVCPKG_USE_OPENCV3=ON"
}

if($ForceOpenCVVersion -gt 0) {
  Move-Item $PSCustomScriptRoot/vcpkg.json $PSCustomScriptRoot/vcpkg.json.bak
  Move-Item $PSCustomScriptRoot/vcpkg.json.opencv23 $PSCustomScriptRoot/vcpkg.json
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

if (-Not $DisableDLLcopy) {
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DX_VCPKG_APPLOCAL_DEPS_INSTALL=ON"
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

if (-Not $InstallDARKNETthroughVCPKG) {
  $AdditionalBuildSetup = $AdditionalBuildSetup + " -DENABLE_DEPLOY_CUSTOM_CMAKE_MODULES:BOOL=ON"
}

if($UseVCPKG) {
  if ($ForceVCPKGBuildtreesPath -ne "") {
    $AdditionalBuildSetup = $AdditionalBuildSetup + " -DVCPKG_INSTALL_OPTIONS=--clean-buildtrees-after-build;--x-buildtrees-root=`"$ForceVCPKGBuildtreesPath`""
    New-Item -Path $ForceVCPKGBuildtreesPath -ItemType directory -Force | Out-Null
    $vcpkgbuildtreespath = "$ForceVCPKGBuildtreesPath"
  }
  else {
    $AdditionalBuildSetup = $AdditionalBuildSetup + " -DVCPKG_INSTALL_OPTIONS=--clean-buildtrees-after-build"
    $vcpkgbuildtreespath = "$vcpkg_path/buildtrees"
  }
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
    Move-Item $PSCustomScriptRoot/vcpkg.json $PSCustomScriptRoot/vcpkg.json.bak
    Push-Location ${env:VCPKG_ROOT}
    if ($ForceVCPKGDarknetHEAD) {
      $proc = Start-Process -NoNewWindow -PassThru -FilePath "${env:VCPKG_ROOT}/vcpkg${exe_ext}" -ArgumentList " remove darknet --recurse "
      $handle = $proc.Handle
      $proc.WaitForExit()
      $exitCode = $proc.ExitCode
      if (-Not ($exitCode -eq 0)) {
        MyThrow("Removing darknet through vcpkg failed! Exited with error code $exitCode.")
      }
    }
    $proc = Start-Process -NoNewWindow -PassThru -FilePath "${env:VCPKG_ROOT}/vcpkg${exe_ext}" -ArgumentList "  upgrade --no-dry-run "
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Upgrading vcpkg installed ports failed! Exited with error code $exitCode.")
    }
    $proc = Start-Process -NoNewWindow -PassThru -FilePath "${env:VCPKG_ROOT}/vcpkg${exe_ext}" -ArgumentList " install darknet[${features}] $headMode --recurse "  # "-manifest"  disables the manifest feature, so that if vcpkg is a subfolder of darknet, the vcpkg.json inside darknet folder does not trigger errors due to automatic manifest mode
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Installing darknet dependencies through vcpkg failed! Exited with error code $exitCode.")
    }
    Move-Item $PSCustomScriptRoot/vcpkg.json.bak $PSCustomScriptRoot/vcpkg.json
  }
}
else {
  if ($BuildDebug -and $UseVCPKG) {
    $debug_build_folder = "$PSCustomScriptRoot/build_debug"
    if (-Not $DoNotDeleteBuildFolder) {
      Write-Host "Removing folder $debug_build_folder" -ForegroundColor Yellow
      Remove-Item -Force -Recurse -ErrorAction SilentlyContinue $debug_build_folder
    }

    New-Item -Path $debug_build_folder -ItemType directory -Force | Out-Null
    Set-Location $debug_build_folder
    $cmake_args = "-G `"$generator`" ${DebugBuildSetup} ${AdditionalBuildSetup} -S .."
    Write-Host "Configuring debug CMake project" -ForegroundColor Green
    Write-Host "CMake args: $cmake_args"
    $proc = Start-Process -NoNewWindow -PassThru -FilePath $CMAKE_EXE -ArgumentList $cmake_args
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Config failed! Exited with error code $exitCode.")
    }
    Write-Host "Building debug CMake project" -ForegroundColor Green
    $proc = Start-Process -NoNewWindow -PassThru -FilePath $CMAKE_EXE -ArgumentList "--build . ${debugConfig} --parallel ${NumberOfBuildWorkers} --target install"
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Build failed! Exited with error code $exitCode.")
    }
    Get-ChildItem "${debug_build_folder}" -Filter *.pdb |
    Foreach-Object {
      Write-Host "-- Copying $_ to $DebugInstallPrefix/bin"
      Copy-Item $_ $DebugInstallPrefix/bin
    }
  }
  $release_build_folder = "$PSCustomScriptRoot/build_release"
  if (-Not $DoNotDeleteBuildFolder) {
    Write-Host "Removing folder $release_build_folder" -ForegroundColor Yellow
    Remove-Item -Force -Recurse -ErrorAction SilentlyContinue $release_build_folder
  }

  New-Item -Path $release_build_folder -ItemType directory -Force | Out-Null
  Set-Location $release_build_folder
  $cmake_args = "-G `"$generator`" ${ReleaseBuildSetup} ${AdditionalBuildSetup} -S .."
  Write-Host "Configuring release CMake project" -ForegroundColor Green
  Write-Host "CMake args: $cmake_args"
  $proc = Start-Process -NoNewWindow -PassThru -FilePath $CMAKE_EXE -ArgumentList $cmake_args
  $handle = $proc.Handle
  $proc.WaitForExit()
  $exitCode = $proc.ExitCode
  if (-Not ($exitCode -eq 0)) {
    MyThrow("Config failed! Exited with error code $exitCode.")
  }
  Write-Host "Building release CMake project" -ForegroundColor Green
  if ($BuildInstaller) {
    $proc = Start-Process -NoNewWindow -PassThru -FilePath $CMAKE_EXE -ArgumentList "--build . ${releaseConfig} --parallel ${NumberOfBuildWorkers}"
  }
  else {
    $proc = Start-Process -NoNewWindow -PassThru -FilePath $CMAKE_EXE -ArgumentList "--build . ${releaseConfig} --parallel ${NumberOfBuildWorkers} --target install"
  }
  $handle = $proc.Handle
  $proc.WaitForExit()
  $exitCode = $proc.ExitCode
  if (-Not ($exitCode -eq 0)) {
    MyThrow("Build failed! Exited with error code $exitCode.")
  }
  if (-Not $UseVCPKG -And -Not $DisableDLLcopy) {
    $dllfiles = Get-ChildItem ./${dllfolder}/*.dll
    if ($dllfiles) {
      Copy-Item $dllfiles ..
    }
  }
  if ($BuildInstaller) {
    Write-Host "Building package with CPack" -ForegroundColor Green
    $proc = Start-Process -NoNewWindow -PassThru -FilePath $CMAKE_EXE -ArgumentList "--build . --target package"
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Packaging failed! Exited with error code $exitCode.")
    }
  }
}

Pop-Location
Write-Host "Build complete!" -ForegroundColor Green

if ($ForceVCPKGBuildtreesRemoval -and (-Not $UseVCPKG)) {
  Write-Host "VCPKG is not enabled, so local vcpkg buildtrees folder will not be deleted even if requested" -ForegroundColor Yellow
}

if ($UseVCPKG -and $ForceVCPKGBuildtreesRemoval) {
  Write-Host "Removing local vcpkg buildtrees folder from $vcpkgbuildtreespath" -ForegroundColor Yellow
  Remove-Item -Force -Recurse -ErrorAction SilentlyContinue $vcpkgbuildtreespath
}

if ($ForceVCPKGPackagesRemoval -and (-Not $UseVCPKG)) {
  Write-Host "VCPKG is not enabled, so local vcpkg packages folder will not be deleted even if requested" -ForegroundColor Yellow
}

if ($UseVCPKG -and $ForceVCPKGPackagesRemoval) {
  $vcpkgpackagespath = "$vcpkg_path/packages"
  Write-Host "Removing local vcpkg packages folder from $vcpkgpackagespath" -ForegroundColor Yellow
  Remove-Item -Force -Recurse -ErrorAction SilentlyContinue $vcpkgpackagespath
}

if ($DownloadWeights) {
  Write-Host "Downloading weights..." -ForegroundColor Yellow
  & $PSScriptRoot/scripts/download_weights.ps1
  Write-Host "Weights downloaded" -ForegroundColor Green
}

if ($vcpkg_root_set_by_this_script) {
  $env:VCPKG_ROOT = $null
}
if ($vcpkg_triplet_set_by_this_script) {
  $env:VCPKG_DEFAULT_TRIPLET = $null
}
if ($vcpkg_host_triplet_set_by_this_script) {
  $env:VCPKG_DEFAULT_HOST_TRIPLET = $null
}

if($ForceOpenCVVersion -gt 0) {
  Move-Item $PSCustomScriptRoot/vcpkg.json $PSCustomScriptRoot/vcpkg.json.opencv23
  Move-Item $PSCustomScriptRoot/vcpkg.json.bak $PSCustomScriptRoot/vcpkg.json
}

if ($vcpkg_branch_set_by_this_script) {
  Push-Location $vcpkg_path
  $git_args = "checkout -"
  $proc = Start-Process -NoNewWindow -PassThru -FilePath $GIT_EXE -ArgumentList "$git_args"
  $handle = $proc.Handle
  $proc.WaitForExit()
  $exitCode = $proc.ExitCode
  if (-Not ($exitCode -eq 0)) {
    MyThrow("Checking out previous branch failed! Exited with error code $exitCode.")
  }
  if ($VCPKGFork -ne "") {
    $git_args = "remote rm vcpkgfork"
    $proc = Start-Process -NoNewWindow -PassThru -FilePath $GIT_EXE -ArgumentList "$git_args"
    $handle = $proc.Handle
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    if (-Not ($exitCode -eq 0)) {
      MyThrow("Checking out previous branch failed! Exited with error code $exitCode.")
    }
  }
  Pop-Location
}

$ErrorActionPreference = "SilentlyContinue"
Stop-Transcript | out-null
$ErrorActionPreference = "Continue"
