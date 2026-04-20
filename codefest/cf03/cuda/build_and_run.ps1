$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat'
$nvcc   = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe'
$dir    = 'C:\Users\Husai\Ece-410-hardware-ai\codefest\cf03\cuda'

# RTX 3070 Ti = Ampere SM 8.6 — generate native SASS only, no PTX embedding
# This avoids driver JIT incompatibility between CUDA 13.2 toolkit and driver 581.95
$arch_flags = '-gencode arch=compute_86,code=sm_86'

# Absorb vcvarsall environment into current PowerShell session
$envLines = cmd /c "`"$vcvars`" x64 && set" 2>&1
foreach ($line in $envLines) {
    if ($line -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}

Set-Location $dir

Write-Host "=== Compiling gemm_naive.cu (sm_86 SASS only) ==="
$out = & "$nvcc" -O2 -gencode arch=compute_86,code=sm_86 -o "$dir\gemm_naive" "$dir\gemm_naive.cu" 2>&1
$out | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -eq 0) { Write-Host "COMPILE OK: gemm_naive.exe" }
else { Write-Host "COMPILE FAILED (exit $LASTEXITCODE)"; exit 1 }

Write-Host ""
Write-Host "=== Compiling gemm_tiled.cu (sm_86 SASS only) ==="
$out = & "$nvcc" -O2 -gencode arch=compute_86,code=sm_86 -o "$dir\gemm_tiled" "$dir\gemm_tiled.cu" 2>&1
$out | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -eq 0) { Write-Host "COMPILE OK: gemm_tiled.exe" }
else { Write-Host "COMPILE FAILED (exit $LASTEXITCODE)"; exit 1 }

Write-Host ""
Write-Host "============================================================"
Write-Host "=== Running gemm_naive.exe ==="
Write-Host "============================================================"
& "$dir\gemm_naive.exe"

Write-Host ""
Write-Host "============================================================"
Write-Host "=== Running gemm_tiled.exe ==="
Write-Host "============================================================"
& "$dir\gemm_tiled.exe"
