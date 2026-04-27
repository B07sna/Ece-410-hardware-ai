// mac_llm_B.v — buggy MAC unit illustrating common LLM-generated mistakes.
//
// Bug 1: always @(posedge clk) instead of always_ff.
//        always_ff signals synthesizer intent (flip-flop only); plain always
//        allows latches and combinational statements, weakening lint/synthesis
//        checks and causing mismatches in some tools.
//
// Bug 2: inputs a and b declared without the signed keyword.
//        The 8×8 multiply is therefore treated as unsigned, so negative
//        values are not sign-extended into the 32-bit accumulator.
//        e.g. a=-5 (8'b11111011=251), b=2 → product=502 instead of -10.
//
// Bug 3: output out declared without signed.
//        The accumulator bits are correct at the register level but the
//        port is typed unsigned, so any downstream signed arithmetic or
//        $signed comparisons on out will interpret it incorrectly.

module mac_B (
    input  logic        clk,
    input  logic        rst,
    input  logic [7:0]  a,          // Bug 2: missing signed
    input  logic [7:0]  b,          // Bug 2: missing signed
    output logic [31:0] out         // Bug 3: missing signed
);

    always @(posedge clk) begin     // Bug 1: should be always_ff
        if (rst)
            out <= 32'h0;
        else
            out <= out + (a * b);   // unsigned multiply; no sign extension
    end

endmodule
