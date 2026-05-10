#!/bin/bash
export PATH=/mingw64/bin:/usr/local/bin:/usr/bin:/bin:$PATH

HDL="/c/Users/Husai/Ece-410-hardware-ai/codefest/cf06/hdl"
SIM="/c/Users/Husai/Ece-410-hardware-ai/codefest/cf06/sim"

echo "=== iverilog compilation ==="
iverilog -g2012 -o "$SIM/crossbar_tb.vvp" \
    "$HDL/crossbar_mac.sv" \
    "$HDL/crossbar_tb.sv"
COMPILE_EXIT=$?
echo "Compile exit code: $COMPILE_EXIT"

if [ $COMPILE_EXIT -eq 0 ]; then
    echo ""
    echo "=== vvp simulation ==="
    vvp "$SIM/crossbar_tb.vvp"
    echo "Sim exit code: $?"
else
    echo "COMPILE FAILED -- simulation not run"
fi
