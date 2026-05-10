`timescale 1ns/1ps
`default_nettype none

// ============================================================
// crossbar_tb.sv  --  self-checking testbench for crossbar_mac
//
// Weight matrix W[i][j]  (row i, col j):
//   row0: [ 1, -1,  1, -1]
//   row1: [ 1,  1, -1, -1]
//   row2: [-1,  1,  1, -1]
//   row3: [-1, -1, -1,  1]
//
// Input vector: in = [10, 20, 30, 40]
//
// Hand-computed expected outputs
//   out[j] = sum_i W[i][j] * in[i]
//
//   out[0] =  1*10 +  1*20 + -1*30 + -1*40 =  10+20-30-40 = -40
//   out[1] = -1*10 +  1*20 +  1*30 + -1*40 = -10+20+30-40 =   0
//   out[2] =  1*10 + -1*20 +  1*30 + -1*40 =  10-20+30-40 = -20
//   out[3] = -1*10 + -1*20 + -1*30 +  1*40 = -10-20-30+40 = -20
// ============================================================

module crossbar_tb;

    // ----------------------------------------------------------------
    // Signals
    // ----------------------------------------------------------------
    logic               clk, rst;
    logic signed [7:0]  in  [3:0];
    logic signed [31:0] out [3:0];
    logic               weight_load;
    logic [3:0]         weight_addr;
    logic               weight_data;

    // ----------------------------------------------------------------
    // DUT
    // ----------------------------------------------------------------
    crossbar_mac dut (
        .clk         (clk),
        .rst         (rst),
        .in          (in),
        .out         (out),
        .weight_load (weight_load),
        .weight_addr (weight_addr),
        .weight_data (weight_data)
    );

    // ----------------------------------------------------------------
    // 10 ns clock
    // ----------------------------------------------------------------
    initial clk = 1'b0;
    always  #5 clk = ~clk;

    // ----------------------------------------------------------------
    // Task: drive one weight cell at the next negedge.
    //   The DUT captures it on the following posedge.
    //   wval: 0 => +1 weight,  1 => -1 weight
    //   weight_addr[3:2] = row,  weight_addr[1:0] = col
    // ----------------------------------------------------------------
    task automatic load_w (
        input [1:0] row,
        input [1:0] col,
        input logic wval
    );
        @(negedge clk);
        weight_load = 1'b1;
        weight_addr = {row, col};
        weight_data = wval;
    endtask

    // ----------------------------------------------------------------
    // Hand-computed expected results
    // ----------------------------------------------------------------
    localparam signed [31:0] EXP0 = -40;
    localparam signed [31:0] EXP1 =   0;
    localparam signed [31:0] EXP2 = -20;
    localparam signed [31:0] EXP3 = -20;

    integer pass_cnt, fail_cnt;

    // ----------------------------------------------------------------
    // Stimulus and self-check
    // ----------------------------------------------------------------
    initial begin
        // Initialise
        rst         = 1'b1;
        weight_load = 1'b0;
        weight_addr = 4'h0;
        weight_data = 1'b0;
        in[0] = 8'sd0;  in[1] = 8'sd0;  in[2] = 8'sd0;  in[3] = 8'sd0;
        pass_cnt = 0;
        fail_cnt = 0;

        // Hold reset for 3 rising edges then release at negedge
        repeat(3) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;

        // ---- Load all 16 weights serially, one per cycle ----

        // Row 0: [ 1, -1,  1, -1]
        load_w(2'd0, 2'd0, 1'b0);   // W[0][0] =  1  (data 0 => +1)
        load_w(2'd0, 2'd1, 1'b1);   // W[0][1] = -1  (data 1 => -1)
        load_w(2'd0, 2'd2, 1'b0);   // W[0][2] =  1
        load_w(2'd0, 2'd3, 1'b1);   // W[0][3] = -1

        // Row 1: [ 1,  1, -1, -1]
        load_w(2'd1, 2'd0, 1'b0);   // W[1][0] =  1
        load_w(2'd1, 2'd1, 1'b0);   // W[1][1] =  1
        load_w(2'd1, 2'd2, 1'b1);   // W[1][2] = -1
        load_w(2'd1, 2'd3, 1'b1);   // W[1][3] = -1

        // Row 2: [-1,  1,  1, -1]
        load_w(2'd2, 2'd0, 1'b1);   // W[2][0] = -1
        load_w(2'd2, 2'd1, 1'b0);   // W[2][1] =  1
        load_w(2'd2, 2'd2, 1'b0);   // W[2][2] =  1
        load_w(2'd2, 2'd3, 1'b1);   // W[2][3] = -1

        // Row 3: [-1, -1, -1,  1]
        load_w(2'd3, 2'd0, 1'b1);   // W[3][0] = -1
        load_w(2'd3, 2'd1, 1'b1);   // W[3][1] = -1
        load_w(2'd3, 2'd2, 1'b1);   // W[3][2] = -1
        load_w(2'd3, 2'd3, 1'b0);   // W[3][3] =  1

        // Deassert weight_load after last capture posedge
        @(negedge clk);
        weight_load = 1'b0;

        // ---- Apply input vector: in = [10, 20, 30, 40] ----
        @(negedge clk);
        in[0] = 8'sd10;
        in[1] = 8'sd20;
        in[2] = 8'sd30;
        in[3] = 8'sd40;

        // One rising edge for the output register to latch acc
        @(posedge clk);
        #1;  // propagation settle

        // ---- Print and verify ----
        // Note: iverilog does not propagate unpacked-array output port values
        // to the connected testbench net; read through hierarchical references
        // (dut.out[j]) which correctly reflect the DUT-internal register.
        $display("============================================");
        $display("  crossbar_mac  4x4 binary-weight MAC TB");
        $display("============================================");
        $display("  W[i][j]  (row i, col j):");
        $display("    row0: [ 1, -1,  1, -1]");
        $display("    row1: [ 1,  1, -1, -1]");
        $display("    row2: [-1,  1,  1, -1]");
        $display("    row3: [-1, -1, -1,  1]");
        $display("  in = [10, 20, 30, 40]");
        $display("  out[j] = sum_i W[i][j]*in[i]");
        $display("--------------------------------------------");

        // out[0]: 1*10 + 1*20 - 1*30 - 1*40 = -40
        if ($signed(dut.out[0]) === EXP0) begin
            $display("  out[0]: expected=%4d  got=%4d  --> PASS", EXP0, $signed(dut.out[0]));
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  out[0]: expected=%4d  got=%4d  --> FAIL", EXP0, $signed(dut.out[0]));
            fail_cnt = fail_cnt + 1;
        end

        // out[1]: -1*10 + 1*20 + 1*30 - 1*40 = 0
        if ($signed(dut.out[1]) === EXP1) begin
            $display("  out[1]: expected=%4d  got=%4d  --> PASS", EXP1, $signed(dut.out[1]));
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  out[1]: expected=%4d  got=%4d  --> FAIL", EXP1, $signed(dut.out[1]));
            fail_cnt = fail_cnt + 1;
        end

        // out[2]: 1*10 - 1*20 + 1*30 - 1*40 = -20
        if ($signed(dut.out[2]) === EXP2) begin
            $display("  out[2]: expected=%4d  got=%4d  --> PASS", EXP2, $signed(dut.out[2]));
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  out[2]: expected=%4d  got=%4d  --> FAIL", EXP2, $signed(dut.out[2]));
            fail_cnt = fail_cnt + 1;
        end

        // out[3]: -1*10 - 1*20 - 1*30 + 1*40 = -20
        if ($signed(dut.out[3]) === EXP3) begin
            $display("  out[3]: expected=%4d  got=%4d  --> PASS", EXP3, $signed(dut.out[3]));
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  out[3]: expected=%4d  got=%4d  --> FAIL", EXP3, $signed(dut.out[3]));
            fail_cnt = fail_cnt + 1;
        end

        $display("--------------------------------------------");
        if (fail_cnt == 0)
            $display("  RESULT: ALL %0d/4 PASSED --> PASS", pass_cnt);
        else
            $display("  RESULT: %0d/4 FAILED     --> FAIL", fail_cnt);
        $display("============================================");

        $finish;
    end

    // Watchdog: abort if stuck
    initial begin
        #5000;
        $display("[TIMEOUT] simulation did not complete in 5000 ns");
        $finish;
    end

endmodule

`default_nettype wire
