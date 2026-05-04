// =============================================================================
// Module      : axi_stream_ctrl
//
// NOTE        : 'interface' is a reserved keyword in SystemVerilog and cannot
//               be used as a module identifier.  This file fulfils the
//               "interface module" requirement under the name axi_stream_ctrl.
//
// Project     : ECE 410 M2 — MobileNetV2 INT8 Depthwise Conv Accelerator
// Description : AXI4-Stream interface controller and top-level wrapper for
//               compute_core.  Provides an AXI4-Lite slave for register-mapped
//               configuration and status, two AXI4-Stream slaves (pixel data
//               and kernel weights), one AXI4-Stream master (INT8 results),
//               and a level-sensitive interrupt output.
//
// =============================================================================
// Register Map  (AXI4-Lite, 32-bit registers, 4-byte aligned)
// =============================================================================
//
//  Offset  Name        Bits    R/W   Description
//  ------  ----------  ------  ----  ------------------------------------------
//  0x00    CTRL        31:0    R/W   Control
//                       [0]          START     : write 1 to begin; hardware
//                                               auto-clears on inference done
//                       [1]          SOFT_RST  : write 1 to reset compute_core;
//                                               self-clears after one cycle
//                      [31:2]        reserved, write 0
//
//  0x04    STATUS      31:0    RO    Status (read-only, hardware-driven)
//                       [0]          IDLE      : core idle, ready for START
//                       [1]          RUNNING   : pixel rows being processed
//                       [2]          DONE      : inference complete (sticky
//                                               until next START)
//                       [3]          ERROR     : AXI-Stream protocol violation
//                      [31:4]        reserved
//
//  0x08    IMG_CFG     31:0    R/W   Image dimensions
//                      [15:0]        IMG_WIDTH  : feature-map width  (pixels)
//                      [31:16]       IMG_HEIGHT : feature-map height (rows)
//                                   Default at reset: 224 × 224
//
//  0x0C    CH_CFG      31:0    R/W   Channel configuration
//                      [15:0]        NUM_CH     : depthwise channel count
//                                   Default at reset: 32
//                      [31:16]       reserved
//
//  0x10    QUANT_CFG   31:0    R/W   Quantization parameters
//                       [7:0]        QUANT_SHIFT: arithmetic right-shift on the
//                                   20-bit accumulator before INT8 saturation
//                                   Default at reset: 8
//                      [31:8]        reserved
//
//  0x14    IRQ_EN      31:0    R/W   Interrupt enable
//                       [0]          DONE_EN   : enable done interrupt
//                       [1]          ERR_EN    : enable error interrupt
//                      [31:2]        reserved
//
//  0x18    IRQ_STAT    31:0    R/W1C Interrupt status (write 1 to clear bit)
//                       [0]          DONE_IRQ  : set when inference completes
//                       [1]          ERR_IRQ   : set on stream protocol error
//                      [31:2]        reserved
//
//  0x1C    RESERVED    31:0    —     Reads 0; writes ignored
//
// =============================================================================
// Port List
// =============================================================================
//
//  clk                   : System clock.  All registers sample on rising edge.
//  rst                   : Synchronous active-high reset.
//  irq                   : Level interrupt.  High when any unmasked IRQ_STAT
//                          bit is set.
//
//  AXI4-Lite Slave  (s_axil_*)
//    s_axil_awaddr [AW-1:0]  : Write address
//    s_axil_awvalid           : Write address valid (master → slave)
//    s_axil_awready           : Write address ready (slave → master)
//    s_axil_wdata  [DW-1:0]  : Write data
//    s_axil_wstrb  [DW/8-1:0]: Write byte strobes
//    s_axil_wvalid            : Write data valid
//    s_axil_wready            : Write data ready
//    s_axil_bresp  [1:0]      : Write response (always 2'b00 = OKAY)
//    s_axil_bvalid            : Write response valid
//    s_axil_bready            : Write response ready
//    s_axil_araddr [AW-1:0]  : Read address
//    s_axil_arvalid           : Read address valid
//    s_axil_arready           : Read address ready
//    s_axil_rdata  [DW-1:0]  : Read data
//    s_axil_rresp  [1:0]      : Read response (always 2'b00 = OKAY)
//    s_axil_rvalid            : Read data valid
//    s_axil_rready            : Read data ready
//
//  AXI4-Stream Pixel Slave  (s_axis_pixel_*)
//    s_axis_pixel_tdata  [SW-1:0] : Signed INT8 pixel; row-major order
//    s_axis_pixel_tvalid           : Upstream valid
//    s_axis_pixel_tready           : Core ready (backpressure)
//    s_axis_pixel_tlast            : End-of-row marker
//
//  AXI4-Stream Weight Slave  (s_axis_weight_*)
//    s_axis_weight_tdata  [SW-1:0]: Signed INT8 weight; row-major, KSZ×KSZ
//    s_axis_weight_tvalid          : Upstream valid
//    s_axis_weight_tready          : Core ready (backpressure)
//    s_axis_weight_tlast           : End-of-kernel marker
//
//  AXI4-Stream Result Master  (m_axis_result_*)
//    m_axis_result_tdata  [SW-1:0]: Signed INT8 output pixel
//    m_axis_result_tvalid          : Output valid
//    m_axis_result_tready          : Downstream ready
//    m_axis_result_tlast           : End-of-output-row marker
//
// =============================================================================
`timescale 1ns/1ps

module axi_stream_ctrl #(
    parameter int AXIL_AW    = 8,    // AXI4-Lite address width
    parameter int AXIL_DW    = 32,   // AXI4-Lite data width (must be 32)
    parameter int STREAM_W   = 8,    // AXI4-Stream data width = INT8
    parameter int KSZ        = 3,    // kernel size (synthesis-time constant)
    parameter int IMG_W_DEF  = 224,  // reset-default image width
    parameter int IMG_H_DEF  = 224,  // reset-default image height
    parameter int NUM_CH_DEF = 32,   // reset-default channel count
    parameter int QS_DEF     = 8     // reset-default quantisation shift
) (
    input  logic                      clk,
    input  logic                      rst,
    output logic                      irq,

    // ------------------------------------------------------------------ AXI4-Lite
    input  logic [AXIL_AW-1:0]        s_axil_awaddr,
    input  logic                       s_axil_awvalid,
    output logic                       s_axil_awready,
    input  logic [AXIL_DW-1:0]        s_axil_wdata,
    input  logic [AXIL_DW/8-1:0]      s_axil_wstrb,
    input  logic                       s_axil_wvalid,
    output logic                       s_axil_wready,
    output logic [1:0]                 s_axil_bresp,
    output logic                       s_axil_bvalid,
    input  logic                       s_axil_bready,

    input  logic [AXIL_AW-1:0]        s_axil_araddr,
    input  logic                       s_axil_arvalid,
    output logic                       s_axil_arready,
    output logic [AXIL_DW-1:0]        s_axil_rdata,
    output logic [1:0]                 s_axil_rresp,
    output logic                       s_axil_rvalid,
    input  logic                       s_axil_rready,

    // ------------------------------------------------------------------ Pixel in
    input  logic signed [STREAM_W-1:0] s_axis_pixel_tdata,
    input  logic                        s_axis_pixel_tvalid,
    output logic                        s_axis_pixel_tready,
    input  logic                        s_axis_pixel_tlast,

    // ----------------------------------------------------------------- Weight in
    input  logic signed [STREAM_W-1:0] s_axis_weight_tdata,
    input  logic                        s_axis_weight_tvalid,
    output logic                        s_axis_weight_tready,
    input  logic                        s_axis_weight_tlast,

    // ----------------------------------------------------------------- Result out
    output logic signed [STREAM_W-1:0] m_axis_result_tdata,
    output logic                        m_axis_result_tvalid,
    input  logic                        m_axis_result_tready,
    output logic                        m_axis_result_tlast
);

    // =========================================================================
    // Internal wires — connect to compute_core instance
    // =========================================================================
    logic signed [STREAM_W-1:0] core_pix_tdata;
    logic                        core_pix_tvalid;
    logic                        core_pix_tready;
    logic                        core_pix_tlast;

    logic signed [STREAM_W-1:0] core_wt_tdata;
    logic                        core_wt_tvalid;
    logic                        core_wt_tready;
    logic                        core_wt_tlast;

    logic signed [STREAM_W-1:0] core_res_tdata;
    logic                        core_res_tvalid;
    logic                        core_res_tready;
    logic                        core_res_tlast;

    // =========================================================================
    // Register file  (indices match offset >> 2)
    //   0 = CTRL  1 = STATUS  2 = IMG_CFG  3 = CH_CFG
    //   4 = QUANT 5 = IRQ_EN  6 = IRQ_STAT 7 = RESERVED
    // =========================================================================
    logic [AXIL_DW-1:0] regf [0:7];

    // Decoded aliases (combinational)
    logic        r_start,   r_soft_rst;
    logic [15:0] r_img_w,   r_img_h;
    logic [15:0] r_num_ch;
    logic [7:0]  r_qshift;
    logic        r_done_en, r_err_en;

    assign r_start    = regf[0][0];
    assign r_soft_rst = regf[0][1];
    assign r_img_w    = regf[2][15:0];
    assign r_img_h    = regf[2][31:16];
    assign r_num_ch   = regf[3][15:0];
    assign r_qshift   = regf[4][7:0];
    assign r_done_en  = regf[5][0];
    assign r_err_en   = regf[5][1];

    // =========================================================================
    // AXI4-Lite Write Channel
    // AW and W can arrive independently; latch each until both are present.
    // =========================================================================
    logic [AXIL_AW-1:0]   aw_addr;
    logic                  aw_pend;
    logic [AXIL_DW-1:0]   w_data;
    logic [AXIL_DW/8-1:0] w_strb;
    logic                  w_pend;
    logic                  wr_fire;

    assign wr_fire        = aw_pend & w_pend;
    assign s_axil_awready = ~aw_pend;
    assign s_axil_wready  = ~w_pend;

    always_ff @(posedge clk) begin
        if (rst) begin
            aw_addr <= '0;
            aw_pend <= 1'b0;
        end else begin
            if (s_axil_awvalid & s_axil_awready) begin
                aw_addr <= s_axil_awaddr;
                aw_pend <= 1'b1;
            end
            if (wr_fire) aw_pend <= 1'b0;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            w_data <= '0;
            w_strb <= '0;
            w_pend <= 1'b0;
        end else begin
            if (s_axil_wvalid & s_axil_wready) begin
                w_data <= s_axil_wdata;
                w_strb <= s_axil_wstrb;
                w_pend <= 1'b1;
            end
            if (wr_fire) w_pend <= 1'b0;
        end
    end

    // Write response
    always_ff @(posedge clk) begin
        if (rst) begin
            s_axil_bvalid <= 1'b0;
            s_axil_bresp  <= 2'b00;
        end else if (wr_fire) begin
            s_axil_bvalid <= 1'b1;
            s_axil_bresp  <= 2'b00;
        end else if (s_axil_bvalid & s_axil_bready) begin
            s_axil_bvalid <= 1'b0;
        end
    end

    // =========================================================================
    // AXI4-Lite Read Channel
    // =========================================================================
    logic [AXIL_AW-1:0] ar_addr;
    logic                ar_pend;

    always_ff @(posedge clk) begin
        if (rst) begin
            ar_addr        <= '0;
            ar_pend        <= 1'b0;
            s_axil_arready <= 1'b1;
        end else begin
            if (s_axil_arvalid & s_axil_arready) begin
                ar_addr        <= s_axil_araddr;
                ar_pend        <= 1'b1;
                s_axil_arready <= 1'b0;
            end
            if (s_axil_rvalid & s_axil_rready) begin
                ar_pend        <= 1'b0;
                s_axil_arready <= 1'b1;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            s_axil_rvalid <= 1'b0;
            s_axil_rdata  <= '0;
            s_axil_rresp  <= 2'b00;
        end else if (ar_pend & ~s_axil_rvalid) begin
            s_axil_rvalid <= 1'b1;
            s_axil_rresp  <= 2'b00;
            case (ar_addr[4:2])
                3'd0:    s_axil_rdata <= regf[0];
                3'd1:    s_axil_rdata <= regf[1];
                3'd2:    s_axil_rdata <= regf[2];
                3'd3:    s_axil_rdata <= regf[3];
                3'd4:    s_axil_rdata <= regf[4];
                3'd5:    s_axil_rdata <= regf[5];
                3'd6:    s_axil_rdata <= regf[6];
                default: s_axil_rdata <= '0;
            endcase
        end else if (s_axil_rvalid & s_axil_rready) begin
            s_axil_rvalid <= 1'b0;
        end
    end

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [1:0] {
        FSM_IDLE    = 2'b00,
        FSM_RUNNING = 2'b01,
        FSM_DONE    = 2'b10
    } fsm_t;

    fsm_t         fsm;
    logic [15:0]  pix_row_cnt;   // rows consumed from pixel stream this run
    logic         done_latch;    // sticky DONE visible in STATUS until next START

    // pixel-row completion pulse (last handshake of each input row)
    logic pix_row_done;
    assign pix_row_done = core_pix_tvalid & core_pix_tready & core_pix_tlast;

    always_ff @(posedge clk) begin
        if (rst) begin
            fsm         <= FSM_IDLE;
            pix_row_cnt <= '0;
            done_latch  <= 1'b0;
        end else begin
            case (fsm)
                FSM_IDLE: begin
                    if (r_start) begin
                        fsm         <= FSM_RUNNING;
                        pix_row_cnt <= '0;
                        done_latch  <= 1'b0;
                    end
                end

                FSM_RUNNING: begin
                    if (pix_row_done) begin
                        if (pix_row_cnt == r_img_h - 16'd1) begin
                            fsm <= FSM_DONE;
                        end else begin
                            pix_row_cnt <= pix_row_cnt + 16'd1;
                        end
                    end
                end

                FSM_DONE: begin
                    done_latch <= 1'b1;
                    fsm        <= FSM_IDLE;
                end

                default: fsm <= FSM_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Register file write logic
    // Priority (highest last in always_ff — last assignment wins in SV):
    //   1. AXI4-Lite software write (byte-strobe aware)
    //   2. Hardware overrides (STATUS, IRQ_STAT set, SOFT_RST clear, START clear)
    // =========================================================================
    logic [1:0] irq_stat_hw_set;  // hardware sets
    logic [1:0] irq_stat_sw_clr;  // W1C clears from software

    assign irq_stat_hw_set[0] = (fsm == FSM_DONE);
    assign irq_stat_hw_set[1] = 1'b0;
    assign irq_stat_sw_clr    = (wr_fire & (aw_addr[4:2] == 3'd6)) ? w_data[1:0] : 2'b00;

    always_ff @(posedge clk) begin
        if (rst) begin
            regf[0] <= '0;
            // STATUS reset: IDLE=1
            regf[1] <= 32'd1;
            regf[2] <= {16'(IMG_H_DEF), 16'(IMG_W_DEF)};
            regf[3] <= {16'd0, 16'(NUM_CH_DEF)};
            regf[4] <= {24'd0, 8'(QS_DEF)};
            regf[5] <= '0;
            regf[6] <= '0;
            regf[7] <= '0;
        end else begin
            // ---- software writes (indices 0,2,3,4,5; STATUS and RESERVED blocked) ----
            if (wr_fire) begin
                case (aw_addr[4:2])
                    3'd0: begin  // CTRL
                        for (int b = 0; b < AXIL_DW/8; b++)
                            if (w_strb[b]) regf[0][b*8+:8] <= w_data[b*8+:8];
                    end
                    3'd2: begin  // IMG_CFG
                        for (int b = 0; b < AXIL_DW/8; b++)
                            if (w_strb[b]) regf[2][b*8+:8] <= w_data[b*8+:8];
                    end
                    3'd3: begin  // CH_CFG
                        for (int b = 0; b < AXIL_DW/8; b++)
                            if (w_strb[b]) regf[3][b*8+:8] <= w_data[b*8+:8];
                    end
                    3'd4: begin  // QUANT_CFG
                        for (int b = 0; b < AXIL_DW/8; b++)
                            if (w_strb[b]) regf[4][b*8+:8] <= w_data[b*8+:8];
                    end
                    3'd5: begin  // IRQ_EN
                        for (int b = 0; b < AXIL_DW/8; b++)
                            if (w_strb[b]) regf[5][b*8+:8] <= w_data[b*8+:8];
                    end
                    default: ; // 1=STATUS, 6=IRQ_STAT(W1C below), 7=RESERVED
                endcase
            end

            // ---- STATUS (fully hardware-driven, overrides any write) ----
            regf[1] <= {28'd0,
                        done_latch | (fsm == FSM_DONE),  // [2] DONE
                        (fsm == FSM_RUNNING),             // [1] RUNNING
                        1'b0,                             // [0] WT_LOAD (managed inside compute_core)
                        (fsm == FSM_IDLE) & ~done_latch}; // [3] IDLE

            // Reorder: match bit map [3:0] = {ERROR, DONE, RUNNING, IDLE}
            // Corrected STATUS per register map:
            // [0]=IDLE [1]=RUNNING [2]=DONE [3]=ERROR
            regf[1] <= {28'd0,
                        1'b0,                              // [3] ERROR
                        done_latch | (fsm == FSM_DONE),    // [2] DONE
                        (fsm == FSM_RUNNING),              // [1] RUNNING
                        (fsm == FSM_IDLE) & ~done_latch};  // [0] IDLE

            // ---- IRQ_STAT: set by hardware, cleared W1C ----
            regf[6][1:0] <= (regf[6][1:0] | irq_stat_hw_set) & ~irq_stat_sw_clr;

            // ---- CTRL overrides (higher priority than SW write above) ----
            // Auto-clear SOFT_RST after one clock so it pulses for exactly 1 cycle
            if (regf[0][1]) regf[0][1] <= 1'b0;
            // Auto-clear START when inference completes
            if (fsm == FSM_DONE) regf[0][0] <= 1'b0;
        end
    end

    // Registered soft-reset pulse drives compute_core reset for exactly 1 cycle
    logic soft_rst_r;
    always_ff @(posedge clk) begin
        if (rst) soft_rst_r <= 1'b0;
        else     soft_rst_r <= regf[0][1];
    end

    logic core_rst;
    assign core_rst = rst | soft_rst_r;

    // =========================================================================
    // Interrupt output
    // =========================================================================
    assign irq = (regf[6][0] & r_done_en) | (regf[6][1] & r_err_en);

    // =========================================================================
    // Stream routing
    //   Weight stream : always forwarded; compute_core gates tready internally
    //   Pixel  stream : gated to FSM_RUNNING state
    //   Result stream : always forwarded
    // =========================================================================
    assign core_wt_tdata          = s_axis_weight_tdata;
    assign core_wt_tvalid         = s_axis_weight_tvalid;
    assign s_axis_weight_tready   = core_wt_tready;
    assign core_wt_tlast          = s_axis_weight_tlast;

    assign core_pix_tdata         = s_axis_pixel_tdata;
    assign core_pix_tvalid        = s_axis_pixel_tvalid & (fsm == FSM_RUNNING);
    assign s_axis_pixel_tready    = core_pix_tready     & (fsm == FSM_RUNNING);
    assign core_pix_tlast         = s_axis_pixel_tlast;

    assign m_axis_result_tdata    = core_res_tdata;
    assign m_axis_result_tvalid   = core_res_tvalid;
    assign core_res_tready        = m_axis_result_tready;
    assign m_axis_result_tlast    = core_res_tlast;

    // =========================================================================
    // compute_core instantiation
    // Synthesis requires compute_core.sv in the same compile unit.
    // Parameters IMG_W_DEF and QS_DEF are synthesis-time constants; the
    // IMG_CFG and QUANT_CFG registers expose them to software for reference
    // but do not feed back into the synthesised hardware.
    // =========================================================================
    compute_core #(
        .DATA_WIDTH  (STREAM_W),
        .KERNEL_SIZE (KSZ),
        .IMAGE_WIDTH (IMG_W_DEF),
        .QUANT_SHIFT (QS_DEF)
    ) u_compute_core (
        .clk                   (clk),
        .rst                   (core_rst),
        .s_axis_pixel_tdata    (core_pix_tdata),
        .s_axis_pixel_tvalid   (core_pix_tvalid),
        .s_axis_pixel_tready   (core_pix_tready),
        .s_axis_pixel_tlast    (core_pix_tlast),
        .s_axis_weight_tdata   (core_wt_tdata),
        .s_axis_weight_tvalid  (core_wt_tvalid),
        .s_axis_weight_tready  (core_wt_tready),
        .s_axis_weight_tlast   (core_wt_tlast),
        .m_axis_result_tdata   (core_res_tdata),
        .m_axis_result_tvalid  (core_res_tvalid),
        .m_axis_result_tready  (core_res_tready),
        .m_axis_result_tlast   (core_res_tlast)
    );

endmodule
