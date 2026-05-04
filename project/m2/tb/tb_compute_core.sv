`timescale 1ns/1ps
// =============================================================================
// Testbench : tb_compute_core
// DUT       : compute_core (weight-stationary INT8 depthwise conv)
// Strategy  : 6x6 image, 3x3 Gaussian-like kernel representative of
//             MobileNetV2 channel-384 dominant depthwise response.
//             Reference model computed in integer arithmetic; every DUT
//             output is compared and logged.  Final PASS/FAIL printed.
// =============================================================================
module tb_compute_core;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    localparam int DW  = 8;               // INT8
    localparam int KSZ = 3;
    localparam int IW  = 6;               // image width  (small for sim speed)
    localparam int IH  = 6;               // image height
    localparam int QS  = 4;               // quant shift  (divide by 16)
    localparam int OW  = IW - KSZ + 1;   // 4
    localparam int OH  = IH - KSZ + 1;   // 4

    // -------------------------------------------------------------------------
    // Clock
    // -------------------------------------------------------------------------
    logic clk = 0;
    always #5 clk = ~clk;   // 100 MHz

    // -------------------------------------------------------------------------
    // DUT ports
    // -------------------------------------------------------------------------
    logic rst;
    logic signed [DW-1:0] s_axis_pixel_tdata;
    logic                  s_axis_pixel_tvalid;
    logic                  s_axis_pixel_tready;
    logic                  s_axis_pixel_tlast;

    logic signed [DW-1:0] s_axis_weight_tdata;
    logic                  s_axis_weight_tvalid;
    logic                  s_axis_weight_tready;
    logic                  s_axis_weight_tlast;

    logic signed [DW-1:0] m_axis_result_tdata;
    logic                  m_axis_result_tvalid;
    logic                  m_axis_result_tready;
    logic                  m_axis_result_tlast;

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    compute_core #(
        .DATA_WIDTH  (DW),
        .KERNEL_SIZE (KSZ),
        .IMAGE_WIDTH (IW),
        .QUANT_SHIFT (QS)
    ) dut (
        .clk                   (clk),
        .rst                   (rst),
        .s_axis_pixel_tdata    (s_axis_pixel_tdata),
        .s_axis_pixel_tvalid   (s_axis_pixel_tvalid),
        .s_axis_pixel_tready   (s_axis_pixel_tready),
        .s_axis_pixel_tlast    (s_axis_pixel_tlast),
        .s_axis_weight_tdata   (s_axis_weight_tdata),
        .s_axis_weight_tvalid  (s_axis_weight_tvalid),
        .s_axis_weight_tready  (s_axis_weight_tready),
        .s_axis_weight_tlast   (s_axis_weight_tlast),
        .m_axis_result_tdata   (m_axis_result_tdata),
        .m_axis_result_tvalid  (m_axis_result_tvalid),
        .m_axis_result_tready  (m_axis_result_tready),
        .m_axis_result_tlast   (m_axis_result_tlast)
    );

    // -------------------------------------------------------------------------
    // Stimulus arrays
    // -------------------------------------------------------------------------
    logic signed [DW-1:0] tb_weights [0:8];
    logic signed [DW-1:0] tb_image   [0:IH-1][0:IW-1];
    logic signed [DW-1:0] tb_expect  [0:OH-1][0:OW-1];

    // -------------------------------------------------------------------------
    // Scoreboard state (shared between two initial blocks)
    // -------------------------------------------------------------------------
    integer pass_cnt;
    integer fail_cnt;
    integer out_row;
    integer out_col;
    integer log_fd;

    // -------------------------------------------------------------------------
    // Reference model — pure integer signed arithmetic
    // Kernel: Gaussian-like 3x3, sum=16 -> shift 4 keeps outputs in INT8
    // Image:  row-major ramp 1..36 (all positive, no saturation needed)
    // -------------------------------------------------------------------------
    integer ref_acc;
    integer ref_sh;
    integer px_val;
    integer wt_val;

    initial begin
        // Kernel weights (MobileNetV2 ch-384 representative, Gaussian-shaped)
        tb_weights[0] = 8'sd1;  tb_weights[1] = 8'sd2;  tb_weights[2] = 8'sd1;
        tb_weights[3] = 8'sd2;  tb_weights[4] = 8'sd4;  tb_weights[5] = 8'sd2;
        tb_weights[6] = 8'sd1;  tb_weights[7] = 8'sd2;  tb_weights[8] = 8'sd1;

        // Feature-map: row-major ramp 1..36, fits comfortably in signed INT8
        for (int r = 0; r < IH; r++)
            for (int c = 0; c < IW; c++)
                tb_image[r][c] = 8'(r * IW + c + 1);

        // Compute reference outputs with integer arithmetic
        for (int r = 0; r < OH; r++) begin
            for (int c = 0; c < OW; c++) begin
                ref_acc = 0;
                for (int kr = 0; kr < KSZ; kr++) begin
                    for (int kc = 0; kc < KSZ; kc++) begin
                        px_val   = $signed(tb_image[r+kr][c+kc]);
                        wt_val   = $signed(tb_weights[kr*KSZ+kc]);
                        ref_acc  = ref_acc + (px_val * wt_val);
                    end
                end
                ref_sh = ref_acc >>> QS;
                if      (ref_sh >  127) tb_expect[r][c] = 8'sh7F;
                else if (ref_sh < -128) tb_expect[r][c] = 8'sh80;
                else                    tb_expect[r][c] = ref_sh[7:0];
            end
        end
    end

    // -------------------------------------------------------------------------
    // Output checker (runs concurrently with stimulus)
    // Samples one cycle after posedge to avoid NBA race with DUT FFs.
    // -------------------------------------------------------------------------
    initial begin
        pass_cnt = 0;
        fail_cnt = 0;
        out_row  = 0;
        out_col  = 0;

        forever begin
            @(posedge clk); #1;
            if (m_axis_result_tvalid && m_axis_result_tready) begin
                if (out_row < OH && out_col < OW) begin
                    if (m_axis_result_tdata === tb_expect[out_row][out_col]) begin
                        $fdisplay(log_fd, "  [%0d][%0d]  got=%4d  exp=%4d  PASS",
                            out_row, out_col,
                            $signed(m_axis_result_tdata),
                            $signed(tb_expect[out_row][out_col]));
                        pass_cnt = pass_cnt + 1;
                    end else begin
                        $fdisplay(log_fd, "  [%0d][%0d]  got=%4d  exp=%4d  FAIL ***",
                            out_row, out_col,
                            $signed(m_axis_result_tdata),
                            $signed(tb_expect[out_row][out_col]));
                        $display("FAIL [%0d][%0d] got=%0d exp=%0d",
                            out_row, out_col,
                            $signed(m_axis_result_tdata),
                            $signed(tb_expect[out_row][out_col]));
                        fail_cnt = fail_cnt + 1;
                    end
                end
                // Advance position using DUT tlast to track row boundaries
                if (m_axis_result_tlast) begin
                    out_row = out_row + 1;
                    out_col = 0;
                end else begin
                    out_col = out_col + 1;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Main stimulus
    // -------------------------------------------------------------------------
    initial begin
        // Open log file
        log_fd = $fopen(
            "C:/Users/Husai/Ece-410-hardware-ai/project/m2/sim/compute_core_run.log",
            "w");
        if (log_fd == 0) begin
            $display("ERROR: cannot open log file");
            $finish;
        end

        $fdisplay(log_fd, "=== tb_compute_core ===");
        $fdisplay(log_fd, "Image %0dx%0d  Kernel %0dx%0d  QUANT_SHIFT=%0d",
            IW, IH, KSZ, KSZ, QS);
        $fdisplay(log_fd, "Kernel (row-major): %0d %0d %0d | %0d %0d %0d | %0d %0d %0d",
            $signed(tb_weights[0]), $signed(tb_weights[1]), $signed(tb_weights[2]),
            $signed(tb_weights[3]), $signed(tb_weights[4]), $signed(tb_weights[5]),
            $signed(tb_weights[6]), $signed(tb_weights[7]), $signed(tb_weights[8]));
        $fdisplay(log_fd, "");
        $fdisplay(log_fd, "Reference outputs:");
        for (int r = 0; r < OH; r++) begin
            for (int c = 0; c < OW; c++)
                $fdisplay(log_fd, "  exp[%0d][%0d] = %0d", r, c,
                    $signed(tb_expect[r][c]));
        end
        $fdisplay(log_fd, "");
        $fdisplay(log_fd, "Per-output comparison:");

        // ---- initialise signals ----
        rst                  = 1;
        s_axis_pixel_tvalid  = 0;
        s_axis_pixel_tdata   = 0;
        s_axis_pixel_tlast   = 0;
        s_axis_weight_tvalid = 0;
        s_axis_weight_tdata  = 0;
        s_axis_weight_tlast  = 0;
        m_axis_result_tready = 1;   // always accept results

        // ---- reset ----
        repeat (4) @(posedge clk);
        #1; rst = 0;

        // ---- Phase 1: load 9 weights, one per clock ----
        // s_axis_weight_tready = ~weights_loaded is always 1 here; no stall check needed.
        $fdisplay(log_fd, "[SIM] Loading weights...");
        for (int i = 0; i < 9; i++) begin
            @(posedge clk); #1;
            s_axis_weight_tvalid = 1;
            s_axis_weight_tdata  = tb_weights[i];
            s_axis_weight_tlast  = (i == 8) ? 1'b1 : 1'b0;
        end
        // De-assert after the last weight is captured
        @(posedge clk); #1;
        s_axis_weight_tvalid = 0;
        s_axis_weight_tlast  = 0;

        // ---- Phase 2: stream 6x6 pixel image row-by-row ----
        // s_axis_pixel_tready rises once weights_loaded=1 (one cycle after last weight).
        // m_axis_result_tready=1 so no output stall; tready stays high throughout.
        $fdisplay(log_fd, "[SIM] Streaming %0dx%0d pixels...", IW, IH);
        for (int r = 0; r < IH; r++) begin
            for (int c = 0; c < IW; c++) begin
                @(posedge clk); #1;
                s_axis_pixel_tvalid = 1;
                s_axis_pixel_tdata  = tb_image[r][c];
                s_axis_pixel_tlast  = (c == IW - 1) ? 1'b1 : 1'b0;
            end
        end
        // One extra posedge to capture the very last pixel, then de-assert
        @(posedge clk); #1;
        s_axis_pixel_tvalid = 0;
        s_axis_pixel_tlast  = 0;

        // ---- drain pipeline (5 stages + output register = 6 cycles; pad to 20) ----
        repeat (20) @(posedge clk);

        // ---- print summary ----
        $fdisplay(log_fd, "");
        $fdisplay(log_fd, "=== Summary ===");
        $fdisplay(log_fd, "Total outputs expected : %0d", OH * OW);
        $fdisplay(log_fd, "PASS                  : %0d", pass_cnt);
        $fdisplay(log_fd, "FAIL                  : %0d", fail_cnt);
        if (fail_cnt == 0)
            $fdisplay(log_fd, "STATUS: PASS");
        else
            $fdisplay(log_fd, "STATUS: FAIL");

        $display("=== tb_compute_core ===");
        $display("PASS %0d / %0d", pass_cnt, OH * OW);
        $display("FAIL %0d / %0d", fail_cnt, OH * OW);
        if (fail_cnt == 0)
            $display("STATUS: PASS");
        else
            $display("STATUS: FAIL");

        $fclose(log_fd);
        $finish;
    end

endmodule
