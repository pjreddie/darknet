#!/usr/bin/env pwsh

$install_cuda = $false

if ($null -eq (Get-Command "choco.exe" -ErrorAction SilentlyContinue)) {
  # Download and install Chocolatey
  Set-ExecutionPolicy unrestricted -Scope CurrentUser
  Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
  Write-Host "Please close and re-open powershell and then re-run setup.ps1 script"
  Break
}

Start-Process -FilePath "choco" -Verb runAs -ArgumentList " install -y cmake ninja powershell git vscode"
Start-Process -FilePath "choco" -Verb runAs -ArgumentList " install -y visualstudio2019buildtools --package-parameters `"--add Microsoft.VisualStudio.Component.VC.CoreBuildTools --includeRecommended --includeOptional --passive --locale en-US --lang en-US`""

if ($install_cuda) {
  Start-Process -FilePath "choco" -Verb runAs -ArgumentList " install -y cuda"
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

git.exe clone https://github.com/microsoft/vcpkg
Set-Location vcpkg
.\bootstrap-vcpkg.bat -disableMetrics
.\vcpkg.exe install darknet[${features}]:x64-windows
