# Launch from an x64 Native Tools Command Prompt for Visual Studio.
param([switch] $RequireCuda)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path build | Out-Null
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw 'MSVC cl.exe not found. Use an x64 Native Tools Command Prompt.'
}
Push-Location build
try {
    & cl.exe /nologo /O2 /LD /EHsc ..\native\blockdist_cpu.cpp /link /OUT:blockdist_cpu.dll
    if ($LASTEXITCODE -ne 0) { throw 'C++ build failed' }
    if (Get-Command nvcc.exe -ErrorAction SilentlyContinue) {
        & nvcc.exe -O3 --shared --cudart=static -o blockdist_cuda.dll ..\native\blockdist_cuda.cu
        if ($LASTEXITCODE -ne 0) { throw 'CUDA build failed' }
    } elseif ($RequireCuda) {
        throw 'nvcc.exe not found; install the NVIDIA CUDA Toolkit.'
    } else {
        Write-Warning 'nvcc.exe not found; built CPU only.'
    }
} finally { Pop-Location }
Write-Host 'Build complete.'
