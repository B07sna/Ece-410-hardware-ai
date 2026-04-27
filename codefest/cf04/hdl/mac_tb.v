`timescale 1ns/1ps

module mac_tb;

    // ------------------------------------------------------------------ //
    // Shared stimulus signals                                              //
    // ------------------------------------------------------------------ //
    logic        clk;
    logic        rst;
    logic signed [7:0] a;
    logic signed [7:0] b;

    // ------------------------------------------------------------------ //
    // DUT outputs                                                          //
    // ------------------------------------------------------------------ //
    logic signed [31:0] out_A;   // correct   (mac_llm_A)
    logic        [31:0] out_B;   // buggy     (mac_llm_B) — unsigned port

    // ------------------------------------------------------------------ //
    // Instantiate both DUTs                                                //
    // ------------------------------------------------------------------ //
    mac mac_A (
        .clk (clk),
        .rst (rst),
        .a   (a),
        .b   (b),
        .out (out_A)
    );

    // Rename module to avoid collision — both files declare module mac.
    // In simulation, compile mac_llm_B separately and use a per-instance
    // defparam / bind, OR rename the module in mac_llm_B.v to mac_B.
    // Here we assume mac_llm_B.v is compiled with module name mac_B.
    mac_B mac_B_inst (
        .clk (clk),
        .rst (rst),
        .a   (a),
        .b   (b),
        .out (out_B)
    );

    // ------------------------------------------------------------------ //
    // Clock: 10 ns period                                                  //
    // ------------------------------------------------------------------ //
    initial clk = 0;
    always #5 clk = ~clk;

    // ------------------------------------------------------------------ //
    // Task: wait one rising edge, then sample and print outputs            //
    // ------------------------------------------------------------------ //
    task tick;
        @(posedge clk); #1;
        $display("t=%4t ns | rst=%b a=%4d b=%4d | out_A=%6d | out_B=%6d%s",
                 $time, rst, a, b,
                 out_A, $signed(out_B),
                 (out_A !== $signed(out_B)) ? "  <-- MISMATCH" : "");
    endtask

    // ------------------------------------------------------------------ //
    // Stimulus                                                             //
    // ------------------------------------------------------------------ //
    initial begin
        $display("=======================================================");
        $display(" mac_tb: comparing mac_llm_A (correct) vs mac_llm_B (buggy)");
        $display("=======================================================");
        $display("%-8s | %-4s %-4s %-3s | %-8s | %-8s",
                 "time", "a", "b", "rst", "out_A", "out_B");
        $display("-------------------------------------------------------");

        // ---- initialise: synchronous reset for one cycle ---------------
        rst = 1; a = 0; b = 0;
        tick;   // out_A = 0, out_B = 0

        // ---- Phase 1: a=3, b=4 for 3 cycles ---------------------------
        rst = 0; a = 3; b = 4;
        tick;   // out = 0  + 3*4 = 12
        tick;   // out = 12 + 3*4 = 24
        tick;   // out = 24 + 3*4 = 36

        // ---- assert rst for one cycle ----------------------------------
        rst = 1; a = 3; b = 4;   // keep a,b driven; rst clears out
        tick;   // out_A = 0, out_B = 0

        // ---- Phase 2: a=-5, b=2 for 2 cycles --------------------------
        // Correct:  (-5)*2 = -10 each cycle
        // Buggy:    (251)*2 = 502 each cycle  (unsigned multiply)
        rst = 0; a = -5; b = 2;
        tick;   // out_A = -10 ; out_B = 502
        tick;   // out_A = -20 ; out_B = 1004

        $display("-------------------------------------------------------");
        $display("Expected out_A after phase 2: -10, then -20");
        $display("Expected out_B after phase 2:  502, then 1004 (wrong!)");
        $display("=======================================================");
        $finish;
    end

endmodule
