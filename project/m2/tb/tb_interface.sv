`timescale 1ns/1ps
// =============================================================================
// Testbench : tb_interface
// DUT       : axi_stream_ctrl (wraps compute_core)
// Tests
//   1. AXI4-Lite write  : QUANT_CFG (0x10) = 4   — verify BRESP=OKAY
//   2. AXI4-Lite read   : STATUS    (0x04) after reset = 0x1 (IDLE)
//   3. AXI4-Lite read   : IMG_CFG   (0x08) after reset = 0x00060006
//   4. Full 6×6 inference via AXI4-Stream (16 outputs vs reference)
//   5. STATUS after inference = 0x4 (DONE)
//   6. IRQ assertion after enabling DONE_EN
//   7. IRQ_STAT W1C clear → irq deasserts
// =============================================================================
module tb_interface;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    localparam int DW  = 8;
    localparam int KSZ = 3;
    localparam int IW  = 6;
    localparam int IH  = 6;
    localparam int QS  = 4;
    localparam int OW  = IW - KSZ + 1;   // 4
    localparam int OH  = IH - KSZ + 1;   // 4

    // -------------------------------------------------------------------------
    // Clock
    // -------------------------------------------------------------------------
    logic clk = 0;
    always #5 clk = ~clk;

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
    // DUT
    // -------------------------------------------------------------------------
    axi_stream_ctrl #(
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
    // Stimulus arrays
    // -------------------------------------------------------------------------
    logic signed [DW-1:0] tb_weights [0:8];
    logic signed [DW-1:0] tb_image   [0:IH-1][0:IW-1];
    logic signed [DW-1:0] tb_expect  [0:OH-1][0:OW-1];

    integer pass_cnt;
    integer fail_cnt;
    integer out_row;
    integer out_col;
    integer log_fd;

    // task output temporaries (module-level so tasks can assign them)
    logic [1:0]  tmp_resp;
    logic [31:0] tmp_rdata;

    // -------------------------------------------------------------------------
    // Reference model (completes at time 0, no delays)
    // -------------------------------------------------------------------------
    integer ref_acc, ref_sh, px_val, wt_val;

    initial begin
        tb_weights[0]=8'sd1; tb_weights[1]=8'sd2; tb_weights[2]=8'sd1;
        tb_weights[3]=8'sd2; tb_weights[4]=8'sd4; tb_weights[5]=8'sd2;
        tb_weights[6]=8'sd1; tb_weights[7]=8'sd2; tb_weights[8]=8'sd1;

        for (int r = 0; r < IH; r++)
            for (int c = 0; c < IW; c++)
                tb_image[r][c] = 8'(r * IW + c + 1);

        for (int r = 0; r < OH; r++) begin
            for (int c = 0; c < OW; c++) begin
                ref_acc = 0;
                for (int kr = 0; kr < KSZ; kr++)
                    for (int kc = 0; kc < KSZ; kc++) begin
                        px_val  = $signed(tb_image[r+kr][c+kc]);
                        wt_val  = $signed(tb_weights[kr*KSZ+kc]);
                        ref_acc = ref_acc + (px_val * wt_val);
                    end
                ref_sh = ref_acc >>> QS;
                if      (ref_sh >  127) tb_expect[r][c] = 8'sh7F;
                else if (ref_sh < -128) tb_expect[r][c] = 8'sh80;
                else                    tb_expect[r][c] = ref_sh[7:0];
            end
        end
    end

    // -------------------------------------------------------------------------
    // Output checker — runs forever; only fires when tvalid=1
    // Samples 1 ns after posedge to avoid NBA race with DUT FFs.
    // -------------------------------------------------------------------------
    initial begin
        out_row = 0;
        out_col = 0;
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
                        fail_cnt = fail_cnt + 1;
                    end
                end
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
    // AXI4-Lite write task
    // Assumes awready=wready=1 on entry (sequential single-master use).
    // Timeline (each line = one posedge):
    //   A+1ns : present awaddr/awvalid, wdata/wstrb/wvalid
    //   B     : both handshakes fire  →  aw_pend=1, w_pend=1
    //   B+1ns : deassert valids
    //   C     : wr_fire=1  →  register written, bvalid←1
    //   C+1ns : assert bready
    //   D     : bvalid & bready  →  bvalid←0
    //   D+1ns : capture bresp
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
    // Timeline:
    //   A+1ns : present araddr/arvalid
    //   B     : AR handshake  →  ar_pend=1, arready←0
    //   B+1ns : deassert arvalid
    //   C     : ar_pend & ~rvalid  →  rvalid←1, rdata latched
    //   C+1ns : assert rready
    //   D     : rvalid & rready  →  rvalid←0
    //   D+1ns : capture rdata/rresp
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
    initial begin
        log_fd = $fopen(
            "C:/Users/Husai/Ece-410-hardware-ai/project/m2/sim/interface_run.log", "w");
        if (log_fd == 0) begin
            $display("ERROR: cannot open log file");
            $finish;
        end

        pass_cnt = 0;
        fail_cnt = 0;

        $fdisplay(log_fd, "=== tb_interface ===");
        $fdisplay(log_fd, "DUT: axi_stream_ctrl  IMG=%0dx%0d  KSZ=%0d  QS=%0d",
            IW, IH, KSZ, QS);
        $fdisplay(log_fd, "");

        // ---- initialise driven signals ----
        rst                  = 1;
        s_axil_awaddr        = 0;  s_axil_awvalid = 0;
        s_axil_wdata         = 0;  s_axil_wstrb   = 0;  s_axil_wvalid  = 0;
        s_axil_bready        = 0;
        s_axil_araddr        = 0;  s_axil_arvalid = 0;
        s_axil_rready        = 0;
        s_axis_pixel_tdata   = 0;  s_axis_pixel_tvalid  = 0;  s_axis_pixel_tlast  = 0;
        s_axis_weight_tdata  = 0;  s_axis_weight_tvalid = 0;  s_axis_weight_tlast = 0;
        m_axis_result_tready = 1;

        repeat (4) @(posedge clk);
        #1; rst = 0;

        // =====================================================================
        // Test 1 — AXI4-Lite write: QUANT_CFG (0x10) = 4
        // =====================================================================
        $fdisplay(log_fd, "--- Test 1: AXI4-Lite write QUANT_CFG=4 (addr=0x10) ---");
        axil_write(8'h10, 32'h00000004, tmp_resp);
        if (tmp_resp === 2'b00) begin
            $fdisplay(log_fd, "  BRESP                  got=0x%0h  exp=0x0  PASS", tmp_resp);
            pass_cnt = pass_cnt + 1;
        end else begin
            $fdisplay(log_fd, "  BRESP                  got=0x%0h  exp=0x0  FAIL ***", tmp_resp);
            fail_cnt = fail_cnt + 1;
        end

        // =====================================================================
        // Test 2 — AXI4-Lite read: STATUS (0x04) after reset = 0x00000001 (IDLE)
        // =====================================================================
        $fdisplay(log_fd, "--- Test 2: Read STATUS (0x04) ---");
        axil_read(8'h04, tmp_rdata, tmp_resp);
        if (tmp_rdata === 32'h00000001) begin
            $fdisplay(log_fd, "  STATUS                 got=0x%08h  exp=0x00000001  PASS", tmp_rdata);
            pass_cnt = pass_cnt + 1;
        end else begin
            $fdisplay(log_fd, "  STATUS                 got=0x%08h  exp=0x00000001  FAIL ***", tmp_rdata);
            fail_cnt = fail_cnt + 1;
        end

        // =====================================================================
        // Test 3 — AXI4-Lite read: IMG_CFG (0x08) = {H=6,W=6} = 0x00060006
        // =====================================================================
        $fdisplay(log_fd, "--- Test 3: Read IMG_CFG (0x08) ---");
        axil_read(8'h08, tmp_rdata, tmp_resp);
        if (tmp_rdata === 32'h00060006) begin
            $fdisplay(log_fd, "  IMG_CFG                got=0x%08h  exp=0x00060006  PASS", tmp_rdata);
            pass_cnt = pass_cnt + 1;
        end else begin
            $fdisplay(log_fd, "  IMG_CFG                got=0x%08h  exp=0x00060006  FAIL ***", tmp_rdata);
            fail_cnt = fail_cnt + 1;
        end

        // =====================================================================
        // Test 4 — Full 6×6 inference
        // =====================================================================
        $fdisplay(log_fd, "--- Test 4: Full 6x6 inference ---");
        $fdisplay(log_fd, "[SIM] Loading weights...");

        for (int i = 0; i < 9; i++) begin
            @(posedge clk); #1;
            s_axis_weight_tdata  = tb_weights[i];
            s_axis_weight_tvalid = 1'b1;
            s_axis_weight_tlast  = (i == 8) ? 1'b1 : 1'b0;
        end
        @(posedge clk); #1;
        s_axis_weight_tvalid = 1'b0;
        s_axis_weight_tlast  = 1'b0;

        // Write START (CTRL[0]=1); FSM transitions to RUNNING one cycle later
        $fdisplay(log_fd, "[SIM] Writing START...");
        axil_write(8'h00, 32'h00000001, tmp_resp);

        // Two extra cycles to ensure FSM is RUNNING before pixels arrive
        repeat (2) @(posedge clk);

        $fdisplay(log_fd, "[SIM] Streaming %0dx%0d pixels...", IW, IH);
        for (int r = 0; r < IH; r++) begin
            for (int c = 0; c < IW; c++) begin
                @(posedge clk); #1;
                s_axis_pixel_tvalid = 1'b1;
                s_axis_pixel_tdata  = tb_image[r][c];
                s_axis_pixel_tlast  = (c == IW - 1) ? 1'b1 : 1'b0;
            end
        end
        @(posedge clk); #1;
        s_axis_pixel_tvalid = 1'b0;
        s_axis_pixel_tlast  = 1'b0;

        // Drain pipeline (5 stages + output reg = 6 latency; pad to 20)
        repeat (20) @(posedge clk);

        $fdisplay(log_fd, "[SIM] Inference done: pass=%0d fail=%0d", pass_cnt, fail_cnt);

        // =====================================================================
        // Test 5 — STATUS after inference = DONE (0x00000004)
        // =====================================================================
        $fdisplay(log_fd, "--- Test 5: Read STATUS (should be DONE) ---");
        axil_read(8'h04, tmp_rdata, tmp_resp);
        if (tmp_rdata === 32'h00000004) begin
            $fdisplay(log_fd, "  STATUS                 got=0x%08h  exp=0x00000004  PASS", tmp_rdata);
            pass_cnt = pass_cnt + 1;
        end else begin
            $fdisplay(log_fd, "  STATUS                 got=0x%08h  exp=0x00000004  FAIL ***", tmp_rdata);
            fail_cnt = fail_cnt + 1;
        end

        // =====================================================================
        // Test 6 — IRQ: enable DONE_EN (IRQ_EN[0]=1), verify irq asserts
        // =====================================================================
        $fdisplay(log_fd, "--- Test 6: IRQ assertion ---");
        axil_write(8'h14, 32'h00000001, tmp_resp);   // IRQ_EN[0] = DONE_EN
        @(posedge clk); #1;
        if (irq === 1'b1) begin
            $fdisplay(log_fd, "  irq                    got=1  exp=1  PASS");
            pass_cnt = pass_cnt + 1;
        end else begin
            $fdisplay(log_fd, "  irq                    got=0  exp=1  FAIL ***");
            fail_cnt = fail_cnt + 1;
        end

        // =====================================================================
        // Test 7 — W1C clear IRQ_STAT[0]; verify irq deasserts
        // =====================================================================
        $fdisplay(log_fd, "--- Test 7: IRQ_STAT W1C clear ---");
        axil_write(8'h18, 32'h00000001, tmp_resp);   // clear DONE_IRQ
        repeat (2) @(posedge clk);

        axil_read(8'h18, tmp_rdata, tmp_resp);
        if (tmp_rdata === 32'h00000000) begin
            $fdisplay(log_fd, "  IRQ_STAT               got=0x%08h  exp=0x00000000  PASS", tmp_rdata);
            pass_cnt = pass_cnt + 1;
        end else begin
            $fdisplay(log_fd, "  IRQ_STAT               got=0x%08h  exp=0x00000000  FAIL ***", tmp_rdata);
            fail_cnt = fail_cnt + 1;
        end

        @(posedge clk); #1;
        if (irq === 1'b0) begin
            $fdisplay(log_fd, "  irq deasserted         got=0  exp=0  PASS");
            pass_cnt = pass_cnt + 1;
        end else begin
            $fdisplay(log_fd, "  irq deasserted         got=1  exp=0  FAIL ***");
            fail_cnt = fail_cnt + 1;
        end

        // =====================================================================
        // Summary
        // =====================================================================
        $fdisplay(log_fd, "");
        $fdisplay(log_fd, "=== Summary ===");
        $fdisplay(log_fd, "Total checks : %0d", pass_cnt + fail_cnt);
        $fdisplay(log_fd, "PASS         : %0d", pass_cnt);
        $fdisplay(log_fd, "FAIL         : %0d", fail_cnt);
        if (fail_cnt == 0)
            $fdisplay(log_fd, "STATUS: PASS");
        else
            $fdisplay(log_fd, "STATUS: FAIL");

        $display("=== tb_interface ===");
        $display("PASS %0d  FAIL %0d", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("STATUS: PASS");
        else               $display("STATUS: FAIL");

        $fclose(log_fd);
        $finish;
    end

endmodule
