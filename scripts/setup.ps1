## enable or disable installed components

$install_choco = $true
$install_cuda = $true

if ($install_cuda -and -not $install_choco) {
  Write-Host "If you want to install cuda without letting the script install also choco, be sure Chocolatey is already installed"
}

###########################

# Download and install Chocolatey
Set-ExecutionPolicy unrestricted
if ($install_choco) {
  Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
}
choco.exe install -y cmake ninja powershell git vscode
choco-exe install -y visualstudio2019buildtools --package-parameters "--add Microsoft.VisualStudio.Component.VC.CoreBuildTools --includeRecommended --includeOptional --passive --locale en-US --lang en-US"

if ($install_cuda) {
  choco-exe install -y cuda
  $features = "full"
}
else {
  if (-not $null -eq $env:CUDA_PATH) {
    $features = "full"
  }
  else{
    $features = "opencv-base,weights,weights-train"
  }
}

Remove-Item -r $temp_folder
Set-Location ..
Set-Location $vcpkg_folder\
git.exe clone https://github.com/microsoft/vcpkg
Set-Location vcpkg
.\bootstrap-vcpkg.bat -disableMetrics
.\vcpkg.exe install darknet[${features}]:x64-windows
