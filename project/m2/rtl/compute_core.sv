// =============================================================================
// Module      : compute_core
// Project     : ECE 410 M2 — MobileNetV2 INT8 Depthwise Conv Accelerator
// Description : Pipelined 3×3 depthwise convolution compute core for INT8
//               inference.  Implements weight-stationary dataflow: kernel
//               weights are loaded once via the weight AXI4-Stream slave, then
//               input feature-map pixels are streamed in row-major order via
//               the pixel slave.  Results emerge as signed INT8 samples on the
//               result master port after a fixed pipeline latency.
//
//               Convolution is valid-only (no zero-padding); the output spatial
//               size is (H − KSZ+1) × (W − KSZ+1) per channel.
//
// Parameters
//   DATA_WIDTH   : Sample bit-width.  Default 8 (= INT8).
//   KERNEL_SIZE  : Kernel spatial dimension; must be odd.  Default 3.
//   IMAGE_WIDTH  : Input feature-map column count.  Default 224.
//   QUANT_SHIFT  : Arithmetic right-shift applied to the 20-bit accumulator
//                  before saturation to DATA_WIDTH.  Default 8.
//
// Ports
//   clk                    : System clock.  All FFs sample rising edge.
//   rst                    : Synchronous active-high reset.
//
//   s_axis_pixel_tdata     : Signed INT8 input pixel  [DATA_WIDTH-1:0]
//   s_axis_pixel_tvalid    : Upstream valid
//   s_axis_pixel_tready    : Core ready (backpressure output)
//   s_axis_pixel_tlast     : End-of-row strobe (asserted on last pixel in row)
//
//   s_axis_weight_tdata    : Signed INT8 kernel weight [DATA_WIDTH-1:0]
//                            Order: row-major w[0][0] … w[KSZ-1][KSZ-1]
//   s_axis_weight_tvalid   : Upstream valid
//   s_axis_weight_tready   : Core ready (deasserted after all weights loaded)
//   s_axis_weight_tlast    : End-of-kernel strobe
//
//   m_axis_result_tdata    : Signed INT8 output pixel [DATA_WIDTH-1:0]
//   m_axis_result_tvalid   : Output valid
//   m_axis_result_tready   : Downstream ready
//   m_axis_result_tlast    : End-of-output-row strobe
//
// Pipeline (PIPE_LATENCY = 5 register stages after pixel fires)
//   Stage 0 : 9 parallel INT8 × INT8 multipliers  → 16-bit products
//   Stage 1 : Level-1 adder tree (4 pairs + 1)    → 5 × 17-bit sums
//   Stage 2 : Level-2 adder tree (2 pairs + 1)    → 3 × 18-bit sums
//   Stage 3 : Level-3 adder tree (1 pair  + 1)    → 2 × 19-bit sums
//   Stage 4 : Final add + arithmetic right-shift + INT8 saturation → output
//
// Dataflow sequencing
//   1. De-assert rst.
//   2. Stream KSZ*KSZ weights on s_axis_weight (assert tlast on last weight).
//      s_axis_weight_tready is high until all weights are accepted.
//   3. s_axis_pixel_tready goes high; stream pixels row-by-row, asserting
//      tlast on the last pixel of every row.
//   4. Valid results appear on m_axis_result after pipeline warm-up
//      (KSZ-1 rows + KSZ-1 columns of input consumed before first output).
// =============================================================================
`timescale 1ns/1ps

module compute_core #(
    parameter int DATA_WIDTH  = 8,
    parameter int KERNEL_SIZE = 3,
    parameter int IMAGE_WIDTH = 224,
    parameter int QUANT_SHIFT = 8
) (
    input  logic                           clk,
    input  logic                           rst,

    // AXI4-Stream pixel slave
    input  logic signed [DATA_WIDTH-1:0]   s_axis_pixel_tdata,
    input  logic                           s_axis_pixel_tvalid,
    output logic                           s_axis_pixel_tready,
    input  logic                           s_axis_pixel_tlast,

    // AXI4-Stream weight slave
    input  logic signed [DATA_WIDTH-1:0]   s_axis_weight_tdata,
    input  logic                           s_axis_weight_tvalid,
    output logic                           s_axis_weight_tready,
    input  logic                           s_axis_weight_tlast,

    // AXI4-Stream result master
    output logic signed [DATA_WIDTH-1:0]   m_axis_result_tdata,
    output logic                           m_axis_result_tvalid,
    input  logic                           m_axis_result_tready,
    output logic                           m_axis_result_tlast
);

    // =========================================================================
    // Derived constants
    // =========================================================================
    localparam int KSZ      = KERNEL_SIZE;
    localparam int NTAPS    = KSZ * KSZ;          // 9
    localparam int MUL_W    = 2 * DATA_WIDTH;     // 16
    localparam int ACC_W    = MUL_W + 4;          // 20  (ceil_log2(9) = 4 extra)
    localparam int COL_BITS = $clog2(IMAGE_WIDTH + 1);

    // =========================================================================
    // Weight register file
    // =========================================================================
    logic signed [DATA_WIDTH-1:0]     weights [0:NTAPS-1];
    logic [$clog2(NTAPS+1)-1:0]       wt_idx;
    logic                             weights_loaded;

    assign s_axis_weight_tready = ~weights_loaded;

    always_ff @(posedge clk) begin
        if (rst) begin
            wt_idx         <= '0;
            weights_loaded <= 1'b0;
            for (int i = 0; i < NTAPS; i++) weights[i] <= '0;
        end else if (~weights_loaded & s_axis_weight_tvalid) begin
            weights[wt_idx] <= s_axis_weight_tdata;
            if (s_axis_weight_tlast | (wt_idx == NTAPS-1)) begin
                weights_loaded <= 1'b1;
                wt_idx         <= '0;
            end else begin
                wt_idx <= wt_idx + 1'b1;
            end
        end
    end

    // =========================================================================
    // Pixel acceptance (stall entire pipeline when output cannot drain)
    // =========================================================================
    logic out_stall;
    assign out_stall           = m_axis_result_tvalid & ~m_axis_result_tready;
    assign s_axis_pixel_tready = weights_loaded & ~out_stall;

    logic pixel_fire;
    assign pixel_fire = s_axis_pixel_tvalid & s_axis_pixel_tready;

    // =========================================================================
    // Row / column counters
    // =========================================================================
    logic [COL_BITS-1:0] col_cnt;
    logic [15:0]         row_cnt;  // fully completed input rows

    always_ff @(posedge clk) begin
        if (rst) begin
            col_cnt <= '0;
            row_cnt <= '0;
        end else if (pixel_fire) begin
            if (s_axis_pixel_tlast | (col_cnt == IMAGE_WIDTH - 1)) begin
                col_cnt <= '0;
                row_cnt <= row_cnt + 16'd1;
            end else begin
                col_cnt <= col_cnt + 1'b1;
            end
        end
    end

    // =========================================================================
    // Line buffers  (hold KSZ-1 = 2 previous rows)
    //   lb0[col] = pixel at (row n-2, col)
    //   lb1[col] = pixel at (row n-1, col)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] lb0 [0:IMAGE_WIDTH-1];
    logic signed [DATA_WIDTH-1:0] lb1 [0:IMAGE_WIDTH-1];

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < IMAGE_WIDTH; i++) begin
                lb0[i] <= '0;
                lb1[i] <= '0;
            end
        end else if (pixel_fire) begin
            lb0[col_cnt] <= lb1[col_cnt];
            lb1[col_cnt] <= s_axis_pixel_tdata;
        end
    end

    // =========================================================================
    // 3×3 horizontal shift-register window
    //   win[r][c] : r=0 oldest row, r=KSZ-1 current row; c=0 oldest col
    // Combinational win_nxt removes 1-cycle valid offset between fire and data.
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] win     [0:KSZ-1][0:KSZ-1];
    logic signed [DATA_WIDTH-1:0] win_nxt [0:KSZ-1][0:KSZ-1];

    always_comb begin
        for (int r = 0; r < KSZ; r++)
            for (int c = 0; c < KSZ-1; c++)
                win_nxt[r][c] = win[r][c+1];   // shift left
        win_nxt[0][KSZ-1] = lb0[col_cnt];       // row n-2 newest entry
        win_nxt[1][KSZ-1] = lb1[col_cnt];       // row n-1 newest entry
        win_nxt[2][KSZ-1] = s_axis_pixel_tdata; // current pixel
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int r = 0; r < KSZ; r++)
                for (int c = 0; c < KSZ; c++)
                    win[r][c] <= '0;
        end else if (pixel_fire) begin
            for (int r = 0; r < KSZ; r++)
                for (int c = 0; c < KSZ; c++)
                    win[r][c] <= win_nxt[r][c];
        end
    end

    // Window is valid when at least KSZ-1 rows have been completed and at
    // least KSZ-1 columns of the current row have been loaded.
    logic win_valid;
    assign win_valid = pixel_fire
                     & (row_cnt  >= 16'(KSZ - 1))
                     & (col_cnt  >= COL_BITS'(KSZ - 1));

    // =========================================================================
    // Pipeline Stage 0 — 9 parallel multipliers
    // =========================================================================
    logic signed [MUL_W-1:0] p0_prod [0:NTAPS-1];
    logic                     p0_valid, p0_last;

    always_ff @(posedge clk) begin
        if (rst) begin
            p0_valid <= 1'b0;
            p0_last  <= 1'b0;
            for (int i = 0; i < NTAPS; i++) p0_prod[i] <= '0;
        end else if (~out_stall) begin
            p0_valid <= win_valid;
            p0_last  <= win_valid & s_axis_pixel_tlast;
            if (win_valid)
                for (int r = 0; r < KSZ; r++)
                    for (int c = 0; c < KSZ; c++)
                        p0_prod[r*KSZ+c] <= win_nxt[r][c] * weights[r*KSZ+c];
        end
    end

    // =========================================================================
    // Pipeline Stage 1 — level-1 adder tree: 4 pairs + 1 singleton → 17-bit
    // =========================================================================
    logic signed [MUL_W:0] p1_sum [0:4];
    logic                   p1_valid, p1_last;

    always_ff @(posedge clk) begin
        if (rst) begin
            p1_valid <= 1'b0;
            p1_last  <= 1'b0;
            for (int i = 0; i < 5; i++) p1_sum[i] <= '0;
        end else if (~out_stall) begin
            p1_valid  <= p0_valid;
            p1_last   <= p0_last;
            p1_sum[0] <= {p0_prod[0][MUL_W-1], p0_prod[0]} + {p0_prod[1][MUL_W-1], p0_prod[1]};
            p1_sum[1] <= {p0_prod[2][MUL_W-1], p0_prod[2]} + {p0_prod[3][MUL_W-1], p0_prod[3]};
            p1_sum[2] <= {p0_prod[4][MUL_W-1], p0_prod[4]} + {p0_prod[5][MUL_W-1], p0_prod[5]};
            p1_sum[3] <= {p0_prod[6][MUL_W-1], p0_prod[6]} + {p0_prod[7][MUL_W-1], p0_prod[7]};
            p1_sum[4] <= {p0_prod[8][MUL_W-1], p0_prod[8]};
        end
    end

    // =========================================================================
    // Pipeline Stage 2 — level-2 adder tree: 2 pairs + 1 singleton → 18-bit
    // =========================================================================
    logic signed [MUL_W+1:0] p2_sum [0:2];
    logic                     p2_valid, p2_last;

    always_ff @(posedge clk) begin
        if (rst) begin
            p2_valid <= 1'b0;
            p2_last  <= 1'b0;
            for (int i = 0; i < 3; i++) p2_sum[i] <= '0;
        end else if (~out_stall) begin
            p2_valid  <= p1_valid;
            p2_last   <= p1_last;
            p2_sum[0] <= {p1_sum[0][MUL_W], p1_sum[0]} + {p1_sum[1][MUL_W], p1_sum[1]};
            p2_sum[1] <= {p1_sum[2][MUL_W], p1_sum[2]} + {p1_sum[3][MUL_W], p1_sum[3]};
            p2_sum[2] <= {p1_sum[4][MUL_W], p1_sum[4]};
        end
    end

    // =========================================================================
    // Pipeline Stage 3 — level-3 adder tree: 1 pair + 1 singleton → 19-bit
    // =========================================================================
    logic signed [MUL_W+2:0] p3_sum [0:1];
    logic                     p3_valid, p3_last;

    always_ff @(posedge clk) begin
        if (rst) begin
            p3_valid <= 1'b0;
            p3_last  <= 1'b0;
            for (int i = 0; i < 2; i++) p3_sum[i] <= '0;
        end else if (~out_stall) begin
            p3_valid  <= p2_valid;
            p3_last   <= p2_last;
            p3_sum[0] <= {p2_sum[0][MUL_W+1], p2_sum[0]} + {p2_sum[1][MUL_W+1], p2_sum[1]};
            p3_sum[1] <= {p2_sum[2][MUL_W+1], p2_sum[2]};
        end
    end

    // =========================================================================
    // Pipeline Stage 4 — final add + quantise
    // =========================================================================
    localparam signed [ACC_W-1:0] SAT_MAX =  20'sd127;
    localparam signed [ACC_W-1:0] SAT_MIN = -20'sd128;

    logic signed [ACC_W-1:0]      acc;
    logic signed [ACC_W-1:0]      acc_shifted;
    logic signed [DATA_WIDTH-1:0] quant;
    logic                         p4_valid, p4_last;

    always_ff @(posedge clk) begin
        if (rst) begin
            acc      <= '0;
            p4_valid <= 1'b0;
            p4_last  <= 1'b0;
        end else if (~out_stall) begin
            p4_valid <= p3_valid;
            p4_last  <= p3_last;
            acc      <= {p3_sum[0][MUL_W+2], p3_sum[0]} +
                        {p3_sum[1][MUL_W+2], p3_sum[1]};
        end
    end

    assign acc_shifted = acc >>> QUANT_SHIFT;

    // Continuous assign: iverilog supports parameter-based part-selects in
    // assign but not inside always_* processes.
    assign quant = (acc_shifted > SAT_MAX) ? SAT_MAX[DATA_WIDTH-1:0] :
                   (acc_shifted < SAT_MIN) ? SAT_MIN[DATA_WIDTH-1:0] :
                                             acc_shifted[DATA_WIDTH-1:0];

    // =========================================================================
    // AXI4-Stream output register
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            m_axis_result_tdata  <= '0;
            m_axis_result_tvalid <= 1'b0;
            m_axis_result_tlast  <= 1'b0;
        end else if (~out_stall) begin
            m_axis_result_tdata  <= quant;
            m_axis_result_tvalid <= p4_valid;
            m_axis_result_tlast  <= p4_last;
        end
    end

endmodule
