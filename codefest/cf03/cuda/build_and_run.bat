@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

cd /d "C:\Users\Husai\Ece-410-hardware-ai\codefest\cf03\cuda"

echo.
echo === Compiling gemm_naive.cu ===
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" -O2 -o gemm_naive gemm_naive.cu
if %ERRORLEVEL% NEQ 0 (
    echo COMPILE FAILED: gemm_naive
    exit /b 1
)
echo COMPILE OK: gemm_naive.exe

echo.
echo === Compiling gemm_tiled.cu ===
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" -O2 -o gemm_tiled gemm_tiled.cu
if %ERRORLEVEL% NEQ 0 (
    echo COMPILE FAILED: gemm_tiled
    exit /b 1
)
echo COMPILE OK: gemm_tiled.exe

echo.
echo ============================================================
echo === Running gemm_naive.exe ===
echo ============================================================
gemm_naive.exe

echo.
echo ============================================================
echo === Running gemm_tiled.exe ===
echo ============================================================
gemm_tiled.exe
