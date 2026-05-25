`timescale 1ns/1ps
// =============================================================================
// Testbench : tb_top
// DUT       : top  (wraps axi_stream_ctrl → compute_core)
// Strategy  : End-to-end test through host-facing ports ONLY.
//             No direct access to compute_core or axi_stream_ctrl internals.
//
// Test vector
//   Kernel  : 3×3, all weights = +1 (INT8)
//   Image   : 3×3 patch, pixels 1..9 (row-major)
//   Expected: sum = 1+2+3+4+5+6+7+8+9 = 45
//             after QUANT_SHIFT=8: 45 >> 8 = 0  (INT8 result = 0)
//
// Test sequence
//   1. Reset DUT.
//   2. Load 9 weights via s_axis_weight (always-forwarded, pre-START).
//   3. Write CTRL[0]=1 (START) via AXI4-Lite — moves FSM to RUNNING.
//   4. Stream 3 rows × 3 pixels via s_axis_pixel (tlast each row end).
//   5. Drain pipeline (5-stage + output reg = 6 cycles; pad to 30).
//   6. Capture m_axis_result_tdata; compare to expected (0).
//   7. Read STATUS via AXI4-Lite — verify DONE bit is set.
//   8. Print PASS / FAIL.
// =============================================================================
module tb_top;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    localparam int DW  = 8;
    localparam int KSZ = 3;
    localparam int IW  = 3;    // image width  — minimal 3-col patch
    localparam int IH  = 3;    // image height — minimal 3-row patch
    localparam int QS  = 8;    // quantization shift
    localparam int OW  = IW - KSZ + 1;   // 1
    localparam int OH  = IH - KSZ + 1;   // 1

    // Expected output: sum=45, 45>>8=0
    localparam signed [DW-1:0] EXPECTED = 8'sd0;

    // -------------------------------------------------------------------------
    // Clock
    // -------------------------------------------------------------------------
    logic clk = 0;
    always #5 clk = ~clk;   // 100 MHz

    // -------------------------------------------------------------------------
    // DUT ports
    // -------------------------------------------------------------------------
    logic rst;
    logic irq;

    logic [7:0]  s_axil_awaddr;
    logic        s_axil_awvalid;
    logic        s_axil_awready;
    logic [31:0] s_axil_wdata;
    logic [3:0]  s_axil_wstrb;
    logic        s_axil_wvalid;
    logic        s_axil_wready;
    logic [1:0]  s_axil_bresp;
    logic        s_axil_bvalid;
    logic        s_axil_bready;

    logic [7:0]  s_axil_araddr;
    logic        s_axil_arvalid;
    logic        s_axil_arready;
    logic [31:0] s_axil_rdata;
    logic [1:0]  s_axil_rresp;
    logic        s_axil_rvalid;
    logic        s_axil_rready;

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
    top #(
        .AXIL_AW    (8),
        .AXIL_DW    (32),
        .STREAM_W   (DW),
        .KSZ        (KSZ),
        .IMG_W_DEF  (IW),
        .IMG_H_DEF  (IH),
        .NUM_CH_DEF (1),
        .QS_DEF     (QS)
    ) dut (
        .clk                  (clk),
        .rst                  (rst),
        .irq                  (irq),
        .s_axil_awaddr        (s_axil_awaddr),
        .s_axil_awvalid       (s_axil_awvalid),
        .s_axil_awready       (s_axil_awready),
        .s_axil_wdata         (s_axil_wdata),
        .s_axil_wstrb         (s_axil_wstrb),
        .s_axil_wvalid        (s_axil_wvalid),
        .s_axil_wready        (s_axil_wready),
        .s_axil_bresp         (s_axil_bresp),
        .s_axil_bvalid        (s_axil_bvalid),
        .s_axil_bready        (s_axil_bready),
        .s_axil_araddr        (s_axil_araddr),
        .s_axil_arvalid       (s_axil_arvalid),
        .s_axil_arready       (s_axil_arready),
        .s_axil_rdata         (s_axil_rdata),
        .s_axil_rresp         (s_axil_rresp),
        .s_axil_rvalid        (s_axil_rvalid),
        .s_axil_rready        (s_axil_rready),
        .s_axis_pixel_tdata   (s_axis_pixel_tdata),
        .s_axis_pixel_tvalid  (s_axis_pixel_tvalid),
        .s_axis_pixel_tready  (s_axis_pixel_tready),
        .s_axis_pixel_tlast   (s_axis_pixel_tlast),
        .s_axis_weight_tdata  (s_axis_weight_tdata),
        .s_axis_weight_tvalid (s_axis_weight_tvalid),
        .s_axis_weight_tready (s_axis_weight_tready),
        .s_axis_weight_tlast  (s_axis_weight_tlast),
        .m_axis_result_tdata  (m_axis_result_tdata),
        .m_axis_result_tvalid (m_axis_result_tvalid),
        .m_axis_result_tready (m_axis_result_tready),
        .m_axis_result_tlast  (m_axis_result_tlast)
    );

    // -------------------------------------------------------------------------
    // Scoreboard
    // -------------------------------------------------------------------------
    integer pass_cnt;
    integer fail_cnt;
    integer log_fd;

    // result capture (filled by output-checker initial block)
    logic signed [DW-1:0] captured_result;
    logic                  result_captured;

    // -------------------------------------------------------------------------
    // Output checker — samples 1 ns after posedge to avoid NBA races
    // -------------------------------------------------------------------------
    initial begin
        result_captured = 1'b0;
        captured_result = '0;
        forever begin
            @(posedge clk); #1;
            if (m_axis_result_tvalid && m_axis_result_tready && !result_captured) begin
                captured_result = m_axis_result_tdata;
                result_captured = 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // AXI4-Lite write task
    // -------------------------------------------------------------------------
    task automatic axil_write(
        input  logic [7:0]  addr,
        input  logic [31:0] data,
        output logic [1:0]  resp
    );
        @(posedge clk); #1;
        s_axil_awaddr  = addr;  s_axil_awvalid = 1'b1;
        s_axil_wdata   = data;  s_axil_wstrb   = 4'hF;  s_axil_wvalid = 1'b1;
        @(posedge clk); #1;
        s_axil_awvalid = 1'b0;  s_axil_wvalid  = 1'b0;
        @(posedge clk); #1;
        s_axil_bready  = 1'b1;
        @(posedge clk); #1;
        resp           = s_axil_bresp;
        s_axil_bready  = 1'b0;
    endtask

    // -------------------------------------------------------------------------
    // AXI4-Lite read task
    // -------------------------------------------------------------------------
    task automatic axil_read(
        input  logic [7:0]  addr,
        output logic [31:0] data,
        output logic [1:0]  resp
    );
        @(posedge clk); #1;
        s_axil_araddr  = addr;  s_axil_arvalid = 1'b1;
        @(posedge clk); #1;
        s_axil_arvalid = 1'b0;
        @(posedge clk); #1;
        s_axil_rready  = 1'b1;
        @(posedge clk); #1;
        data           = s_axil_rdata;
        resp           = s_axil_rresp;
        s_axil_rready  = 1'b0;
    endtask

    // -------------------------------------------------------------------------
    // Main stimulus
    // -------------------------------------------------------------------------
    logic [1:0]  tmp_resp;
    logic [31:0] tmp_rdata;

    initial begin
        // VCD dump for waveform generation
        $dumpfile("C:/Users/Husai/Ece-410-hardware-ai/project/m3/sim/tb_top.vcd");
        $dumpvars(0, tb_top);

        // Open log
        log_fd = $fopen(
            "C:/Users/Husai/Ece-410-hardware-ai/project/m3/sim/cosim_run.log", "w");
        if (log_fd == 0) begin
            $display("ERROR: cannot open log file"); $finish;
        end

        pass_cnt = 0;
        fail_cnt = 0;

        $fdisplay(log_fd, "=== tb_top — M3 End-to-End Co-Simulation ===");
        $fdisplay(log_fd, "DUT : top -> axi_stream_ctrl -> compute_core");
        $fdisplay(log_fd, "Test: 3x3 all-ones kernel, pixels 1..9, expected output = 0");
        $fdisplay(log_fd, "      (sum=45, QUANT_SHIFT=8 => 45>>8 = 0)");
        $fdisplay(log_fd, "");

        // ---- initialise driven signals ----
        rst                  = 1;
        s_axil_awaddr        = 0; s_axil_awvalid = 0;
        s_axil_wdata         = 0; s_axil_wstrb   = 0; s_axil_wvalid  = 0;
        s_axil_bready        = 0;
        s_axil_araddr        = 0; s_axil_arvalid = 0;
        s_axil_rready        = 0;
        s_axis_pixel_tdata   = 0; s_axis_pixel_tvalid  = 0; s_axis_pixel_tlast  = 0;
        s_axis_weight_tdata  = 0; s_axis_weight_tvalid = 0; s_axis_weight_tlast = 0;
        m_axis_result_tready = 1;

        repeat (4) @(posedge clk);
        #1; rst = 0;

        // =====================================================================
        // Phase 1 — Load 9 weights (all +1) via AXI4-Stream weight slave
        // Weight stream is forwarded regardless of FSM state.
        // =====================================================================
        $fdisplay(log_fd, "--- Phase 1: Load 9 weights (all +1) via s_axis_weight ---");
        for (int i = 0; i < 9; i++) begin
            @(posedge clk); #1;
            s_axis_weight_tdata  = 8'sd1;
            s_axis_weight_tvalid = 1'b1;
            s_axis_weight_tlast  = (i == 8) ? 1'b1 : 1'b0;
        end
        @(posedge clk); #1;
        s_axis_weight_tvalid = 1'b0;
        s_axis_weight_tlast  = 1'b0;
        $fdisplay(log_fd, "  Weights loaded.");

        // =====================================================================
        // Phase 2 — Write START via AXI4-Lite CTRL register (offset 0x00)
        // This moves the FSM from IDLE to RUNNING and gates the pixel stream.
        // =====================================================================
        $fdisplay(log_fd, "--- Phase 2: Write CTRL[0]=1 (START) via AXI4-Lite ---");
        axil_write(8'h00, 32'h00000001, tmp_resp);
        if (tmp_resp === 2'b00) begin
            $fdisplay(log_fd, "  CTRL write BRESP = OKAY  PASS");
            pass_cnt = pass_cnt + 1;
        end else begin
            $fdisplay(log_fd, "  CTRL write BRESP = 0x%0h  FAIL ***", tmp_resp);
            fail_cnt = fail_cnt + 1;
        end

        // Extra cycles to let FSM settle to RUNNING
        repeat (2) @(posedge clk);

        // =====================================================================
        // Phase 3 — Stream 3×3 pixel patch (pixels 1..9) via s_axis_pixel
        // tlast is asserted on pixel 3 (col 2) of each row.
        // =====================================================================
        $fdisplay(log_fd, "--- Phase 3: Stream 3x3 pixels (1..9) via s_axis_pixel ---");
        for (int r = 0; r < IH; r++) begin
            for (int c = 0; c < IW; c++) begin
                @(posedge clk); #1;
                s_axis_pixel_tdata  = 8'(r * IW + c + 1);
                s_axis_pixel_tvalid = 1'b1;
                s_axis_pixel_tlast  = (c == IW - 1) ? 1'b1 : 1'b0;
            end
        end
        @(posedge clk); #1;
        s_axis_pixel_tvalid = 1'b0;
        s_axis_pixel_tlast  = 1'b0;
        $fdisplay(log_fd, "  Pixel streaming complete.");

        // =====================================================================
        // Phase 4 — Drain pipeline (5 stages + output reg; pad to 30 cycles)
        // =====================================================================
        repeat (30) @(posedge clk);

        // =====================================================================
        // Test: compare captured result to expected (0)
        // =====================================================================
        $fdisplay(log_fd, "--- Test: Output comparison ---");
        $fdisplay(log_fd, "  Kernel: all-ones 3x3");
        $fdisplay(log_fd, "  Pixels: 1..9 (sum=45)");
        $fdisplay(log_fd, "  QUANT_SHIFT=%0d  => 45>>%0d = %0d", QS, QS, (45 >>> QS));
        $fdisplay(log_fd, "  Expected result : %4d", $signed(EXPECTED));

        if (result_captured) begin
            $fdisplay(log_fd, "  Captured result : %4d", $signed(captured_result));
            if (captured_result === EXPECTED) begin
                $fdisplay(log_fd, "  Output match    : PASS");
                pass_cnt = pass_cnt + 1;
            end else begin
                $fdisplay(log_fd, "  Output match    : FAIL *** got=%0d exp=%0d",
                    $signed(captured_result), $signed(EXPECTED));
                fail_cnt = fail_cnt + 1;
            end
        end else begin
            $fdisplay(log_fd, "  ERROR: no output received from DUT — FAIL ***");
            fail_cnt = fail_cnt + 1;
        end

        // =====================================================================
        // AXI4-Lite read: STATUS should show DONE (bit[2]=1 => 0x00000004)
        // Wait a few extra cycles for FSM to process the last row completion.
        // =====================================================================
        repeat (5) @(posedge clk);
        $fdisplay(log_fd, "--- AXI4-Lite read STATUS (0x04) ---");
        axil_read(8'h04, tmp_rdata, tmp_resp);
        $fdisplay(log_fd, "  STATUS = 0x%08h", tmp_rdata);
        if (tmp_rdata[2] === 1'b1) begin
            $fdisplay(log_fd, "  DONE bit set     : PASS");
            pass_cnt = pass_cnt + 1;
        end else begin
            $fdisplay(log_fd, "  DONE bit not set : FAIL *** STATUS=0x%08h", tmp_rdata);
            fail_cnt = fail_cnt + 1;
        end

        // =====================================================================
        // Summary
        // =====================================================================
        $fdisplay(log_fd, "");
        $fdisplay(log_fd, "=== Summary ===");
        $fdisplay(log_fd, "Checks PASS : %0d", pass_cnt);
        $fdisplay(log_fd, "Checks FAIL : %0d", fail_cnt);
        if (fail_cnt == 0)
            $fdisplay(log_fd, "STATUS: PASS");
        else
            $fdisplay(log_fd, "STATUS: FAIL");

        $display("=== tb_top ===");
        $display("PASS %0d  FAIL %0d", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("STATUS: PASS");
        else               $display("STATUS: FAIL");

        $fclose(log_fd);
        $finish;
    end

endmodule
