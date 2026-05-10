`default_nettype none

// 4x4 binary-weight crossbar MAC unit
//
// Weight encoding: 1-bit per cell
//   weight[i][j] = 0  =>  +1
//   weight[i][j] = 1  =>  -1
//
// Computation (registered, 1-cycle latency):
//   out[j] = sum_i( weight[i][j] * in[i] )
//
// Weight load interface (serial, one cell per cycle):
//   weight_addr[3:2] = row index i  (0..3)
//   weight_addr[1:0] = col index j  (0..3)
//   weight_data      = 0 (+1) or 1 (-1)
//
// Implementation note: all loop-based accesses to unpacked arrays are
// fully unrolled to work around iverilog's limited support for variable
// indices into unpacked arrays inside always blocks.

module crossbar_mac (
    input  logic               clk,
    input  logic               rst,

    // Activation inputs: 4 signed 8-bit values
    input  logic signed [7:0]  in  [3:0],

    // MAC outputs: 4 signed 32-bit values, registered
    output logic signed [31:0] out [3:0],

    // Weight load interface
    input  logic               weight_load,
    input  logic [3:0]         weight_addr,
    input  logic               weight_data
);

    // ----------------------------------------------------------------
    // 4x4 weight register array: 0 = +1, 1 = -1
    // ----------------------------------------------------------------
    logic weight [3:0][3:0];

    always_ff @(posedge clk) begin
        if (rst) begin
            weight[0][0] <= 1'b0; weight[0][1] <= 1'b0;
            weight[0][2] <= 1'b0; weight[0][3] <= 1'b0;
            weight[1][0] <= 1'b0; weight[1][1] <= 1'b0;
            weight[1][2] <= 1'b0; weight[1][3] <= 1'b0;
            weight[2][0] <= 1'b0; weight[2][1] <= 1'b0;
            weight[2][2] <= 1'b0; weight[2][3] <= 1'b0;
            weight[3][0] <= 1'b0; weight[3][1] <= 1'b0;
            weight[3][2] <= 1'b0; weight[3][3] <= 1'b0;
        end else if (weight_load) begin
            // Case-decode avoids dynamic indexing into unpacked array
            case (weight_addr)
                4'h0: weight[0][0] <= weight_data;
                4'h1: weight[0][1] <= weight_data;
                4'h2: weight[0][2] <= weight_data;
                4'h3: weight[0][3] <= weight_data;
                4'h4: weight[1][0] <= weight_data;
                4'h5: weight[1][1] <= weight_data;
                4'h6: weight[1][2] <= weight_data;
                4'h7: weight[1][3] <= weight_data;
                4'h8: weight[2][0] <= weight_data;
                4'h9: weight[2][1] <= weight_data;
                4'hA: weight[2][2] <= weight_data;
                4'hB: weight[2][3] <= weight_data;
                4'hC: weight[3][0] <= weight_data;
                4'hD: weight[3][1] <= weight_data;
                4'hE: weight[3][2] <= weight_data;
                4'hF: weight[3][3] <= weight_data;
                default: ;
            endcase
        end
    end

    // ----------------------------------------------------------------
    // Sign-extend 8-bit inputs to 32 bits (unrolled continuous assigns)
    // ----------------------------------------------------------------
    logic signed [31:0] in_sx [3:0];

    assign in_sx[0] = {{24{in[0][7]}}, in[0]};
    assign in_sx[1] = {{24{in[1][7]}}, in[1]};
    assign in_sx[2] = {{24{in[2][7]}}, in[2]};
    assign in_sx[3] = {{24{in[3][7]}}, in[3]};

    // ----------------------------------------------------------------
    // Combinational accumulation (fully unrolled)
    // acc[j] = sum_i( weight[i][j] * in_sx[i] )
    // ----------------------------------------------------------------
    logic signed [31:0] acc [3:0];

    always_comb begin
        // Column 0
        acc[0] = 32'sd0;
        acc[0] = weight[0][0] ? acc[0] - in_sx[0] : acc[0] + in_sx[0];
        acc[0] = weight[1][0] ? acc[0] - in_sx[1] : acc[0] + in_sx[1];
        acc[0] = weight[2][0] ? acc[0] - in_sx[2] : acc[0] + in_sx[2];
        acc[0] = weight[3][0] ? acc[0] - in_sx[3] : acc[0] + in_sx[3];
        // Column 1
        acc[1] = 32'sd0;
        acc[1] = weight[0][1] ? acc[1] - in_sx[0] : acc[1] + in_sx[0];
        acc[1] = weight[1][1] ? acc[1] - in_sx[1] : acc[1] + in_sx[1];
        acc[1] = weight[2][1] ? acc[1] - in_sx[2] : acc[1] + in_sx[2];
        acc[1] = weight[3][1] ? acc[1] - in_sx[3] : acc[1] + in_sx[3];
        // Column 2
        acc[2] = 32'sd0;
        acc[2] = weight[0][2] ? acc[2] - in_sx[0] : acc[2] + in_sx[0];
        acc[2] = weight[1][2] ? acc[2] - in_sx[1] : acc[2] + in_sx[1];
        acc[2] = weight[2][2] ? acc[2] - in_sx[2] : acc[2] + in_sx[2];
        acc[2] = weight[3][2] ? acc[2] - in_sx[3] : acc[2] + in_sx[3];
        // Column 3
        acc[3] = 32'sd0;
        acc[3] = weight[0][3] ? acc[3] - in_sx[0] : acc[3] + in_sx[0];
        acc[3] = weight[1][3] ? acc[3] - in_sx[1] : acc[3] + in_sx[1];
        acc[3] = weight[2][3] ? acc[3] - in_sx[2] : acc[3] + in_sx[2];
        acc[3] = weight[3][3] ? acc[3] - in_sx[3] : acc[3] + in_sx[3];
    end

    // ----------------------------------------------------------------
    // Register outputs (unrolled)
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            out[0] <= 32'sd0;
            out[1] <= 32'sd0;
            out[2] <= 32'sd0;
            out[3] <= 32'sd0;
        end else begin
            out[0] <= acc[0];
            out[1] <= acc[1];
            out[2] <= acc[2];
            out[3] <= acc[3];
        end
    end

endmodule

`default_nettype wire
