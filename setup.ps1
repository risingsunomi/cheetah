param(
    [string]$Torch = "",
    [string]$Cuda = "",
    [string]$Python = "",
    [string]$Venv = "",
    [switch]$NoVenv,
    [switch]$Dev,
    [switch]$DryRun,
    [switch]$Help
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
try {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
}
catch {
}

if ($Help) {
    @"
Usage: .\setup.ps1 [options]

Creates a Python virtual environment, installs the correct PyTorch wheel, then
installs Cheetah in editable mode.

Options:
  -Torch TARGET       auto, cpu, cuda, cu118, cu126, cu128, or cu130
  -Cuda VERSION      CUDA version to use for selection, e.g. 12.8 or 13.0
  -Dev               Install Cheetah with its dev extras
  -Python PATH       Python executable to use before the venv is created
  -Venv PATH         Virtual environment directory (default: .venv)
  -NoVenv            Install into the current Python environment
  -DryRun            Print the commands that would run
  -Help              Show this help

Environment overrides:
  CHEETAH_TORCH_TARGET=auto|cpu|cuda|cu118|cu126|cu128|cu130
  CHEETAH_CUDA_VERSION=12.8
  CHEETAH_VENV=.venv
"@
    exit 0
}

if (-not $Torch) {
    if ($env:CHEETAH_TORCH_TARGET) { $Torch = $env:CHEETAH_TORCH_TARGET } else { $Torch = "auto" }
}
if (-not $Cuda) {
    if ($env:CHEETAH_CUDA_VERSION) { $Cuda = $env:CHEETAH_CUDA_VERSION } else { $Cuda = "auto" }
}
if (-not $Venv) {
    if ($env:CHEETAH_VENV) { $Venv = $env:CHEETAH_VENV } else { $Venv = ".venv" }
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host "warning: $Message" -ForegroundColor Yellow
}

function Write-Banner {
    # Logo text mirrors cheetah/tui/main_menu.py.
    $lines = @(
        "░░      ░░░  ░░░░  ░░        ░░        ░░        ░░░      ░░░  ░░░░  ░",
        "▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒",
        "▓  ▓▓▓▓▓▓▓▓        ▓▓      ▓▓▓▓      ▓▓▓▓▓▓▓  ▓▓▓▓▓  ▓▓▓▓  ▓▓        ▓",
        "█  ████  ██  ████  ██  ████████  ███████████  █████        ██  ████  █",
        "██      ███  ████  ██        ██        █████  █████  ████  ██  ████  █"
    )
    $colors = @("Cyan", "DarkCyan", "Green", "Yellow", "Magenta")

    Write-Host ""
    for ($i = 0; $i -lt $lines.Count; $i++) {
        Write-Host $lines[$i] -ForegroundColor $colors[$i]
    }
    Write-Host "Cheetah installer" -ForegroundColor White
    Write-Host ""
}

function Invoke-Logged {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )
    Write-Step ("+ {0} {1}" -f $FilePath, ($Arguments -join " "))
    if (-not $DryRun) {
        & $FilePath @Arguments
    }
}

function Get-Platform {
    if ($env:OS -eq "Windows_NT") {
        return "windows"
    }
    $isMac = Get-Variable -Name IsMacOS -ErrorAction SilentlyContinue
    if ($isMac -and $IsMacOS) {
        return "macos"
    }
    $isLinux = Get-Variable -Name IsLinux -ErrorAction SilentlyContinue
    if ($isLinux -and $IsLinux) {
        return "linux"
    }
    return "windows"
}

function Get-PythonCommand {
    if ($Python) {
        return @($Python)
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        return @($pyLauncher.Source, "-3")
    }

    foreach ($candidate in @("python3.14", "python3.13", "python3.12", "python3.11", "python3.10", "python3", "python")) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) {
            return @($cmd.Source)
        }
    }

    throw "Python 3.10 or newer was not found. Install Python first, then rerun setup."
}

function Invoke-BasePython {
    param([string[]]$Arguments)
    $exe = $script:BasePython[0]
    $args = @()
    if ($script:BasePython.Count -gt 1) {
        $args += $script:BasePython[1..($script:BasePython.Count - 1)]
    }
    $args += $Arguments
    Invoke-Logged $exe $args
}

function Test-BasePythonVersion {
    if ($DryRun) {
        return
    }
    $exe = $script:BasePython[0]
    $args = @()
    if ($script:BasePython.Count -gt 1) {
        $args += $script:BasePython[1..($script:BasePython.Count - 1)]
    }
    $args += @("-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 'Python 3.10 or newer is required; found ' + sys.version.split()[0])")
    & $exe @args
}

function Invoke-VenvPython {
    param([string[]]$Arguments)
    $args = @()
    if ($script:PyPrefixArgs.Count -gt 0) {
        $args += $script:PyPrefixArgs
    }
    $args += $Arguments
    Invoke-Logged $script:PyExe $args
}

function Invoke-PythonCapture {
    param([string[]]$Arguments)
    $args = @()
    if ($script:PyPrefixArgs.Count -gt 0) {
        $args += $script:PyPrefixArgs
    }
    $args += $Arguments
    return (& $script:PyExe @args)
}

function Install-PipPackage {
    param([string[]]$Arguments)
    Invoke-VenvPython (@("-m", "pip", "install") + $Arguments)
}

function Get-CudaVersion {
    if ($Cuda -and $Cuda -ne "auto") {
        return $Cuda
    }

    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $output = & $nvidiaSmi.Source 2>$null
        foreach ($line in $output) {
            if ($line -match "CUDA Version:\s*([0-9]+(\.[0-9]+)?)") {
                return $Matches[1]
            }
        }
    }

    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        $output = & $nvcc.Source --version 2>$null
        foreach ($line in $output) {
            if ($line -match "release\s+([0-9]+(\.[0-9]+)?)") {
                return $Matches[1]
            }
        }
    }

    return ""
}

function Convert-VersionNumber {
    param([string]$Version)
    if ($Version -match "^([0-9]+)(?:\.([0-9]+))?") {
        $major = [int]$Matches[1]
        $minor = 0
        if ($Matches[2]) {
            $minor = [int]$Matches[2]
        }
        return ($major * 100 + $minor)
    }
    return 0
}

function Select-CudaTag {
    param([string]$Version)
    $number = Convert-VersionNumber $Version
    if ($number -ge 1300) { return "cu130" }
    if ($number -ge 1208) { return "cu128" }
    if ($number -ge 1206) { return "cu126" }
    if ($number -ge 1108) { return "cu118" }
    return ""
}

function Resolve-TorchVariant {
    $requested = $Torch.ToLowerInvariant().Replace(".", "").Replace("-", "_")
    switch ($requested) {
        "auto" {
            $cudaVersion = Get-CudaVersion
            if ($cudaVersion) {
                $tag = Select-CudaTag $cudaVersion
                if ($tag) { return $tag }
                Write-Warn "CUDA $cudaVersion was detected, but supported PyTorch wheel tags are cu118, cu126, cu128, cu130. Falling back to CPU."
            }
            return "cpu"
        }
        "cuda" {
            $cudaVersion = Get-CudaVersion
            if (-not $cudaVersion) {
                throw "CUDA was requested but no CUDA version could be detected. Use -Cuda 12.8 or choose -Torch cpu."
            }
            $tag = Select-CudaTag $cudaVersion
            if (-not $tag) {
                throw "CUDA $cudaVersion is below the supported wheel floor. Use -Torch cpu or install a newer NVIDIA driver."
            }
            return $tag
        }
        "cpu" { return "cpu" }
        "cu118" { return "cu118" }
        "cu126" { return "cu126" }
        "cu128" { return "cu128" }
        "cu130" { return "cu130" }
        "118" { return "cu118" }
        "11_8" { return "cu118" }
        "cu11_8" { return "cu118" }
        "cuda118" { return "cu118" }
        "cuda_118" { return "cu118" }
        "cuda11_8" { return "cu118" }
        "126" { return "cu126" }
        "12_6" { return "cu126" }
        "cu12_6" { return "cu126" }
        "cuda126" { return "cu126" }
        "cuda_126" { return "cu126" }
        "cuda12_6" { return "cu126" }
        "128" { return "cu128" }
        "12_8" { return "cu128" }
        "cu12_8" { return "cu128" }
        "cuda128" { return "cu128" }
        "cuda_128" { return "cu128" }
        "cuda12_8" { return "cu128" }
        "130" { return "cu130" }
        "13_0" { return "cu130" }
        "cu13_0" { return "cu130" }
        "cuda130" { return "cu130" }
        "cuda_130" { return "cu130" }
        "cuda13_0" { return "cu130" }
        default {
            throw "unsupported torch target '$Torch'. Use auto, cpu, cuda, cu118, cu126, cu128, or cu130."
        }
    }
}

function Get-TorchIndexUrl {
    param([string]$Variant)
    switch ($Variant) {
        "cu118" { return "https://download.pytorch.org/whl/cu118" }
        "cu126" { return "https://download.pytorch.org/whl/cu126" }
        "cu128" { return "https://download.pytorch.org/whl/cu128" }
        "cu130" { return "https://download.pytorch.org/whl/cu130" }
        "cpu" { return "https://download.pytorch.org/whl/cpu" }
        default { return "" }
    }
}

function Install-Torch {
    param([string]$Variant)
    $packages = @("torch", "torchvision", "torchaudio")
    $args = @("--upgrade") + $packages
    $indexUrl = Get-TorchIndexUrl $Variant
    if ($indexUrl) {
        $args += @("--index-url", $indexUrl)
    }
    Install-PipPackage $args
}

function Get-InstalledTorchBaseVersion {
    if ($DryRun) {
        if ($env:TC_TORCH_VERSION) { return $env:TC_TORCH_VERSION }
        return "2.10.0"
    }
    return (Invoke-PythonCapture @("-c", "import torch; print(torch.__version__.split('+', 1)[0])"))
}

function Get-InstalledTorchLocalTag {
    if ($DryRun) {
        return ""
    }
    return (Invoke-PythonCapture @("-c", "import torch; v=torch.__version__; print(v.split('+', 1)[1] if '+' in v else '')"))
}

function Install-Project {
    param(
        [string]$Variant,
        [string]$TorchVersion
    )

    $spec = "."
    if ($Dev) {
        $spec = ".[dev]"
    }

    Write-Step "Installing Cheetah with TC_TORCH_VARIANT=$Variant and TC_TORCH_VERSION=$TorchVersion"
    if ($DryRun) {
        Write-Step "+ TC_TORCH_VARIANT=$Variant TC_TORCH_VERSION=$TorchVersion $script:PyExe -m pip install -e $spec"
        return
    }

    $oldVariant = $env:TC_TORCH_VARIANT
    $oldVersion = $env:TC_TORCH_VERSION
    try {
        $env:TC_TORCH_VARIANT = $Variant
        $env:TC_TORCH_VERSION = $TorchVersion
        Invoke-VenvPython @("-m", "pip", "install", "-e", $spec)
    }
    finally {
        $env:TC_TORCH_VARIANT = $oldVariant
        $env:TC_TORCH_VERSION = $oldVersion
    }
}

function Test-Install {
    if ($DryRun) {
        return
    }
    Invoke-PythonCapture @("-c", "import cheetah, torch; print('cheetah import: ok'); print('torch: ' + torch.__version__); print('torch.cuda.is_available: ' + str(torch.cuda.is_available()))")
}

$Platform = Get-Platform
Write-Banner

$script:BasePython = Get-PythonCommand
Test-BasePythonVersion

if ($NoVenv) {
    $script:PyExe = $script:BasePython[0]
    $script:PyPrefixArgs = @()
    if ($script:BasePython.Count -gt 1) {
        $script:PyPrefixArgs = $script:BasePython[1..($script:BasePython.Count - 1)]
    }
}
else {
    if (-not (Test-Path $Venv)) {
        Invoke-BasePython @("-m", "venv", $Venv)
    }
    else {
        Write-Step "Using existing virtual environment: $Venv"
    }

    if ($Platform -eq "windows") {
        $script:PyExe = Join-Path $Venv "Scripts\python.exe"
    }
    else {
        $script:PyExe = Join-Path $Venv "bin/python"
    }
    $script:PyPrefixArgs = @()
}

Write-Step "Platform: $Platform"
if ($script:PyPrefixArgs.Count -gt 0) {
    Write-Step ("Python: {0} {1}" -f $script:PyExe, ($script:PyPrefixArgs -join " "))
}
else {
    Write-Step "Python: $script:PyExe"
}

Install-PipPackage @("--upgrade", "pip", "setuptools", "wheel")

$TorchVariant = Resolve-TorchVariant
Write-Step "Selected PyTorch target: $TorchVariant"

Install-Torch $TorchVariant
$TorchVersion = Get-InstalledTorchBaseVersion
$LocalTag = Get-InstalledTorchLocalTag

if ($TorchVariant.StartsWith("cu") -and $LocalTag -and $LocalTag -ne $TorchVariant) {
    Write-Warn "Installed torch local tag is '$LocalTag', but '$TorchVariant' was selected."
}

Install-Project $TorchVariant $TorchVersion
Test-Install

Write-Step "Setup complete."
if (-not $NoVenv) {
    Write-Step "Activate with: $Venv\Scripts\Activate.ps1"
}
