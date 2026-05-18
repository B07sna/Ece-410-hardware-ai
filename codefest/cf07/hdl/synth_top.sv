`default_nettype none

// Yosys synthesis view of crossbar_mac (cf06/hdl/crossbar_mac.sv).
// Unpacked array ports flattened to individual wires; weight storage
// expanded to 16 named registers; always_ff / always_comb replaced with
// standard always blocks.  Functional behaviour is bit-for-bit identical
// to the simulation source.
//
// 4x4 binary-weight crossbar MAC unit
//   weight[i][j] = 0 => +1,  1 => -1
//   out[j] = sum_i( weight[i][j] * in[i] ),  registered 1-cycle latency

module crossbar_mac (
    input  wire        clk,
    input  wire        rst,

    // Activation inputs: 4 signed 8-bit values (flattened from in[3:0])
    input  wire signed [7:0]  in0,
    input  wire signed [7:0]  in1,
    input  wire signed [7:0]  in2,
    input  wire signed [7:0]  in3,

    // MAC outputs: 4 signed 32-bit values, registered (flattened from out[3:0])
    output reg  signed [31:0] out0,
    output reg  signed [31:0] out1,
    output reg  signed [31:0] out2,
    output reg  signed [31:0] out3,

    // Weight load interface
    input  wire        weight_load,
    input  wire [3:0]  weight_addr,
    input  wire        weight_data
);

    // ----------------------------------------------------------------
    // 4x4 weight registers (0 = +1, 1 = -1) — named to avoid unpacked arrays
    // Row index = first digit, col index = second digit
    // ----------------------------------------------------------------
    reg w00, w01, w02, w03;
    reg w10, w11, w12, w13;
    reg w20, w21, w22, w23;
    reg w30, w31, w32, w33;

    always @(posedge clk) begin
        if (rst) begin
            w00 <= 1'b0; w01 <= 1'b0; w02 <= 1'b0; w03 <= 1'b0;
            w10 <= 1'b0; w11 <= 1'b0; w12 <= 1'b0; w13 <= 1'b0;
            w20 <= 1'b0; w21 <= 1'b0; w22 <= 1'b0; w23 <= 1'b0;
            w30 <= 1'b0; w31 <= 1'b0; w32 <= 1'b0; w33 <= 1'b0;
        end else if (weight_load) begin
            case (weight_addr)
                4'h0: w00 <= weight_data;
                4'h1: w01 <= weight_data;
                4'h2: w02 <= weight_data;
                4'h3: w03 <= weight_data;
                4'h4: w10 <= weight_data;
                4'h5: w11 <= weight_data;
                4'h6: w12 <= weight_data;
                4'h7: w13 <= weight_data;
                4'h8: w20 <= weight_data;
                4'h9: w21 <= weight_data;
                4'hA: w22 <= weight_data;
                4'hB: w23 <= weight_data;
                4'hC: w30 <= weight_data;
                4'hD: w31 <= weight_data;
                4'hE: w32 <= weight_data;
                4'hF: w33 <= weight_data;
                default: ;
            endcase
        end
    end

    // ----------------------------------------------------------------
    // Sign-extend 8-bit inputs to 32 bits
    // ----------------------------------------------------------------
    wire signed [31:0] in0_sx = {{24{in0[7]}}, in0};
    wire signed [31:0] in1_sx = {{24{in1[7]}}, in1};
    wire signed [31:0] in2_sx = {{24{in2[7]}}, in2};
    wire signed [31:0] in3_sx = {{24{in3[7]}}, in3};

    // ----------------------------------------------------------------
    // Combinational accumulation (fully unrolled)
    // acc_j = sum_i( w_ij ? -in_i : +in_i )
    // ----------------------------------------------------------------
    reg signed [31:0] acc0, acc1, acc2, acc3;

    always @(*) begin
        // Column 0
        acc0 = 32'sd0;
        acc0 = w00 ? acc0 - in0_sx : acc0 + in0_sx;
        acc0 = w10 ? acc0 - in1_sx : acc0 + in1_sx;
        acc0 = w20 ? acc0 - in2_sx : acc0 + in2_sx;
        acc0 = w30 ? acc0 - in3_sx : acc0 + in3_sx;
        // Column 1
        acc1 = 32'sd0;
        acc1 = w01 ? acc1 - in0_sx : acc1 + in0_sx;
        acc1 = w11 ? acc1 - in1_sx : acc1 + in1_sx;
        acc1 = w21 ? acc1 - in2_sx : acc1 + in2_sx;
        acc1 = w31 ? acc1 - in3_sx : acc1 + in3_sx;
        // Column 2
        acc2 = 32'sd0;
        acc2 = w02 ? acc2 - in0_sx : acc2 + in0_sx;
        acc2 = w12 ? acc2 - in1_sx : acc2 + in1_sx;
        acc2 = w22 ? acc2 - in2_sx : acc2 + in2_sx;
        acc2 = w32 ? acc2 - in3_sx : acc2 + in3_sx;
        // Column 3
        acc3 = 32'sd0;
        acc3 = w03 ? acc3 - in0_sx : acc3 + in0_sx;
        acc3 = w13 ? acc3 - in1_sx : acc3 + in1_sx;
        acc3 = w23 ? acc3 - in2_sx : acc3 + in2_sx;
        acc3 = w33 ? acc3 - in3_sx : acc3 + in3_sx;
    end

    // ----------------------------------------------------------------
    // Register outputs
    // ----------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            out0 <= 32'sd0;
            out1 <= 32'sd0;
            out2 <= 32'sd0;
            out3 <= 32'sd0;
        end else begin
            out0 <= acc0;
            out1 <= acc1;
            out2 <= acc2;
            out3 <= acc3;
        end
    end

endmodule

`default_nettype wire
