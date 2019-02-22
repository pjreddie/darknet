#!/usr/bin/env pwsh

$number_of_build_workers=8
$vstype="Community"

if ((Get-Command "cl.exe" -ErrorAction SilentlyContinue) -eq $null)
{
  pushd "C:\Program Files (x86)\Microsoft Visual Studio\2017\${vstype}\Common7\Tools"
  cmd /c "VsDevCmd.bat -arch=x64 & set" |
    foreach {
    if ($_ -match "=") {
      $v = $_.split("="); set-item -force -path "ENV:\$($v[0])"  -value "$($v[1])"
    }
  }
  popd
  Write-Host "Visual Studio 2017 ${vstype} Command Prompt variables set.`n" -ForegroundColor Yellow
}

if (Test-Path env:VCPKG_ROOT) {
  # DEBUG
  New-Item -Path .\build_win_debug -ItemType directory -Force
  Set-Location build_win_debug
  cmake -G "Visual Studio 15 2017" -T "host=x64" -A "x64" "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" "-DVCPKG_TARGET_TRIPLET=$env:VCPKG_DEFAULT_TRIPLET"   "-DCMAKE_BUILD_TYPE=Debug" ..
  cmake --build . --config Debug --parallel ${number_of_build_workers}
  Set-Location ..

  # RELEASE
  New-Item -Path .\build_win_release -ItemType directory -Force
  Set-Location build_win_release
  cmake -G "Visual Studio 15 2017" -T "host=x64" -A "x64" "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" "-DVCPKG_TARGET_TRIPLET=$env:VCPKG_DEFAULT_TRIPLET"   "-DCMAKE_BUILD_TYPE=Release" ..
  cmake --build . --config Release --parallel ${number_of_build_workers}
  Set-Location ..
}
else {
  Write-Host "Skipping vcpkg-enabled builds because the VCPKG_ROOT environment variable is not defined`n" -ForegroundColor Yellow
}

# USE LOCAL PTHREAD LIB, NO VCPKG: remember to use "vcpkg.exe integrate remove" in case you had enable user-wide vcpkg integration
New-Item -Path .\build_win_release_cuda_int_libs -ItemType directory -Force
Set-Location build_win_release_cuda_int_libs
cmake -G "Visual Studio 15 2017" -T "host=x64" -A "x64" ..
cmake --build . --config Release --parallel ${number_of_build_workers}
Set-Location ..
