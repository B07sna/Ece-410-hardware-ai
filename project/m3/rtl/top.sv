// =============================================================================
// Module      : top
// Project     : ECE 410 M3 — MobileNetV2 INT8 Depthwise Conv Accelerator
// Description : Integrated top-level module.  Wraps axi_stream_ctrl (which
//               internally instantiates compute_core) and exposes all host-
//               facing ports: AXI4-Lite for register control and three
//               AXI4-Stream channels (pixel input, weight input, result output)
//               for data movement.  This shell adds no logic; it exists to
//               provide a clean integration boundary and a single module name
//               for synthesis and top-level simulation.
//
// Parameters
//   AXIL_AW    : AXI4-Lite address bus width.  Default 8.
//   AXIL_DW    : AXI4-Lite data bus width; must be 32.  Default 32.
//   STREAM_W   : AXI4-Stream sample width = INT8.  Default 8.
//   KSZ        : Depthwise convolution kernel spatial dimension.  Default 3.
//   IMG_W_DEF  : Reset-default feature-map width (columns).  Default 224.
//   IMG_H_DEF  : Reset-default feature-map height (rows).  Default 224.
//   NUM_CH_DEF : Reset-default depthwise channel count.  Default 32.
//   QS_DEF     : Reset-default quantization arithmetic-right-shift.  Default 8.
//
// Ports
//   clk                    : System clock.  All registers sample rising edge.
//   rst                    : Synchronous active-high reset.
//   irq                    : Level-sensitive interrupt.  Asserted when any
//                            unmasked IRQ_STAT bit in axi_stream_ctrl is set.
//
//   AXI4-Lite Slave  (host → accelerator register access)
//     s_axil_awaddr        : Write address [AXIL_AW-1:0]
//     s_axil_awvalid       : Write address valid
//     s_axil_awready       : Write address ready
//     s_axil_wdata         : Write data [AXIL_DW-1:0]
//     s_axil_wstrb         : Write byte strobes [AXIL_DW/8-1:0]
//     s_axil_wvalid        : Write data valid
//     s_axil_wready        : Write data ready
//     s_axil_bresp         : Write response [1:0] — always OKAY (2'b00)
//     s_axil_bvalid        : Write response valid
//     s_axil_bready        : Write response ready
//     s_axil_araddr        : Read address [AXIL_AW-1:0]
//     s_axil_arvalid       : Read address valid
//     s_axil_arready       : Read address ready
//     s_axil_rdata         : Read data [AXIL_DW-1:0]
//     s_axil_rresp         : Read response [1:0] — always OKAY (2'b00)
//     s_axil_rvalid        : Read data valid
//     s_axil_rready        : Read data ready
//
//   AXI4-Stream Pixel Slave  (host → accelerator, INT8 feature-map pixels)
//     s_axis_pixel_tdata   : Signed INT8 pixel [STREAM_W-1:0]; row-major order
//     s_axis_pixel_tvalid  : Upstream valid
//     s_axis_pixel_tready  : Core ready — backpressure to host
//     s_axis_pixel_tlast   : End-of-row marker; assert on last pixel of each row
//
//   AXI4-Stream Weight Slave  (host → accelerator, INT8 kernel weights)
//     s_axis_weight_tdata  : Signed INT8 weight [STREAM_W-1:0]; row-major,
//                            w[0][0]..w[KSZ-1][KSZ-1]
//     s_axis_weight_tvalid : Upstream valid
//     s_axis_weight_tready : Core ready — backpressure to host
//     s_axis_weight_tlast  : End-of-kernel marker; assert on last weight
//
//   AXI4-Stream Result Master  (accelerator → host, INT8 conv outputs)
//     m_axis_result_tdata  : Signed INT8 output pixel [STREAM_W-1:0]
//     m_axis_result_tvalid : Output valid
//     m_axis_result_tready : Downstream ready — backpressure from host
//     m_axis_result_tlast  : End-of-output-row marker
//
// Integration notes
//   axi_stream_ctrl contains compute_core as a sub-instance.  All AXI4-Stream
//   routing between the two is handled inside axi_stream_ctrl; no additional
//   wiring is required at this level.  The pixel stream is gated to the
//   FSM_RUNNING state internally; the host must write CTRL[0]=1 (START) via
//   AXI4-Lite before streaming pixels or they will be silently dropped.
// =============================================================================
`timescale 1ns/1ps

module top #(
    parameter int AXIL_AW    = 8,
    parameter int AXIL_DW    = 32,
    parameter int STREAM_W   = 8,
    parameter int KSZ        = 3,
    parameter int IMG_W_DEF  = 224,
    parameter int IMG_H_DEF  = 224,
    parameter int NUM_CH_DEF = 32,
    parameter int QS_DEF     = 8
) (
    input  logic                        clk,
    input  logic                        rst,
    output logic                        irq,

    // -------------------------------------------------------------- AXI4-Lite
    input  logic [AXIL_AW-1:0]         s_axil_awaddr,
    input  logic                        s_axil_awvalid,
    output logic                        s_axil_awready,
    input  logic [AXIL_DW-1:0]         s_axil_wdata,
    input  logic [AXIL_DW/8-1:0]       s_axil_wstrb,
    input  logic                        s_axil_wvalid,
    output logic                        s_axil_wready,
    output logic [1:0]                  s_axil_bresp,
    output logic                        s_axil_bvalid,
    input  logic                        s_axil_bready,

    input  logic [AXIL_AW-1:0]         s_axil_araddr,
    input  logic                        s_axil_arvalid,
    output logic                        s_axil_arready,
    output logic [AXIL_DW-1:0]         s_axil_rdata,
    output logic [1:0]                  s_axil_rresp,
    output logic                        s_axil_rvalid,
    input  logic                        s_axil_rready,

    // ---------------------------------------------------------- Pixel stream
    input  logic signed [STREAM_W-1:0] s_axis_pixel_tdata,
    input  logic                        s_axis_pixel_tvalid,
    output logic                        s_axis_pixel_tready,
    input  logic                        s_axis_pixel_tlast,

    // --------------------------------------------------------- Weight stream
    input  logic signed [STREAM_W-1:0] s_axis_weight_tdata,
    input  logic                        s_axis_weight_tvalid,
    output logic                        s_axis_weight_tready,
    input  logic                        s_axis_weight_tlast,

    // --------------------------------------------------------- Result stream
    output logic signed [STREAM_W-1:0] m_axis_result_tdata,
    output logic                        m_axis_result_tvalid,
    input  logic                        m_axis_result_tready,
    output logic                        m_axis_result_tlast
);

    axi_stream_ctrl #(
        .AXIL_AW    (AXIL_AW),
        .AXIL_DW    (AXIL_DW),
        .STREAM_W   (STREAM_W),
        .KSZ        (KSZ),
        .IMG_W_DEF  (IMG_W_DEF),
        .IMG_H_DEF  (IMG_H_DEF),
        .NUM_CH_DEF (NUM_CH_DEF),
        .QS_DEF     (QS_DEF)
    ) u_ctrl (
        .clk                   (clk),
        .rst                   (rst),
        .irq                   (irq),

        .s_axil_awaddr         (s_axil_awaddr),
        .s_axil_awvalid        (s_axil_awvalid),
        .s_axil_awready        (s_axil_awready),
        .s_axil_wdata          (s_axil_wdata),
        .s_axil_wstrb          (s_axil_wstrb),
        .s_axil_wvalid         (s_axil_wvalid),
        .s_axil_wready         (s_axil_wready),
        .s_axil_bresp          (s_axil_bresp),
        .s_axil_bvalid         (s_axil_bvalid),
        .s_axil_bready         (s_axil_bready),

        .s_axil_araddr         (s_axil_araddr),
        .s_axil_arvalid        (s_axil_arvalid),
        .s_axil_arready        (s_axil_arready),
        .s_axil_rdata          (s_axil_rdata),
        .s_axil_rresp          (s_axil_rresp),
        .s_axil_rvalid         (s_axil_rvalid),
        .s_axil_rready         (s_axil_rready),

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

endmodule
