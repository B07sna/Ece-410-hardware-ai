#!/bin/bash
export PATH=/mingw64/bin:/usr/local/bin:/usr/bin:/bin:$PATH
echo "=== iverilog location ==="
which iverilog 2>/dev/null || echo "iverilog not found in PATH"
echo "=== vvp location ==="
which vvp 2>/dev/null || echo "vvp not found"
echo "=== system.vpi DLL deps ==="
ldd /mingw64/lib/ivl/system.vpi 2>/dev/null || echo "ldd failed"
echo "=== PATH ==="
echo $PATH