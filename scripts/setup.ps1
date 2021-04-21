#!/usr/bin/env pwsh

$install_cuda = $false

if ($null -eq (Get-Command "choco.exe" -ErrorAction SilentlyContinue)) {
  # Download and install Chocolatey
  Set-ExecutionPolicy unrestricted -Scope CurrentUser
  Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
  Throw "Please close and re-open powershell and then re-run setup.ps1 script"
}

Start-Process -FilePath "choco" -Verb runAs -ArgumentList " install -y cmake ninja powershell git vscode"
Start-Process -FilePath "choco" -Verb runAs -ArgumentList " install -y visualstudio2019buildtools --package-parameters `"--add Microsoft.VisualStudio.Component.VC.CoreBuildTools --includeRecommended --includeOptional --passive --locale en-US --lang en-US`""
Push-Location $PSScriptRoot

if ($install_cuda) {
  & ./deploy-cuda.ps1
  $features = "full"
}
else {
  if (-not $null -eq $env:CUDA_PATH) {
    $features = "full"
  }
  else{
    $features = "opencv-base"
  }
}

git.exe clone https://github.com/microsoft/vcpkg ../vcpkg
Set-Location ..\vcpkg
.\bootstrap-vcpkg.bat -disableMetrics
.\vcpkg.exe install darknet[${features}]:x64-windows
Pop-Location

Write-Host "Darknet installed in $pwd\x64-windows\tools\darknet" -ForegroundColor Yellow
