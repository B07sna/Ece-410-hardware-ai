lib_content = r'''/* Generic 130 nm standard-cell library -- educational synthesis
 * Scalar cell_rise/cell_fall in nanoseconds; area in um2 (1 GE = 2 um2).
 * Note: setup/hold arcs on DFF D pins omitted for Yosys 0.52 dfflibmap
 * compatibility (dfflibmap rejects timing() groups on sequential input pins).
 */
library(generic_130nm) {
  time_unit                : "1ns";
  voltage_unit             : "1V";
  current_unit             : "1uA";
  capacitive_load_unit     (1.0, ff);
  leakage_power_unit       : "1nW";

  default_cell_leakage_power : 0.0;
  default_fanout_load        : 1.0;
  default_inout_pin_cap      : 0.0;
  default_input_pin_cap      : 0.01;
  default_output_pin_cap     : 0.0;

  /* -- Inverter --------------------------------------------------------- */
  cell(INV) {
    area : 1.0;
    pin(A) { direction: input;  capacitance: 0.01; }
    pin(Y) {
      direction: output;
      function: "(!A)";
      timing() {
        related_pin: "A";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.15"); }
        rise_transition(scalar){ values("0.04"); }
        cell_fall(scalar)      { values("0.12"); }
        fall_transition(scalar){ values("0.04"); }
      }
    }
  }

  /* -- 2-input NAND ---------------------------------------------------- */
  cell(NAND2) {
    area : 1.5;
    pin(A) { direction: input;  capacitance: 0.01; }
    pin(B) { direction: input;  capacitance: 0.01; }
    pin(Y) {
      direction: output;
      function: "!(A&B)";
      timing() {
        related_pin: "A";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.17"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.13"); }
        fall_transition(scalar){ values("0.04"); }
      }
      timing() {
        related_pin: "B";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.17"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.13"); }
        fall_transition(scalar){ values("0.04"); }
      }
    }
  }

  /* -- 3-input NAND ---------------------------------------------------- */
  cell(NAND3) {
    area : 2.0;
    pin(A) { direction: input;  capacitance: 0.01; }
    pin(B) { direction: input;  capacitance: 0.01; }
    pin(C) { direction: input;  capacitance: 0.01; }
    pin(Y) {
      direction: output;
      function: "!(A&B&C)";
      timing() {
        related_pin: "A";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.22"); }
        rise_transition(scalar){ values("0.06"); }
        cell_fall(scalar)      { values("0.16"); }
        fall_transition(scalar){ values("0.05"); }
      }
      timing() {
        related_pin: "B";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.22"); }
        rise_transition(scalar){ values("0.06"); }
        cell_fall(scalar)      { values("0.16"); }
        fall_transition(scalar){ values("0.05"); }
      }
      timing() {
        related_pin: "C";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.22"); }
        rise_transition(scalar){ values("0.06"); }
        cell_fall(scalar)      { values("0.16"); }
        fall_transition(scalar){ values("0.05"); }
      }
    }
  }

  /* -- 2-input NOR ----------------------------------------------------- */
  cell(NOR2) {
    area : 1.5;
    pin(A) { direction: input;  capacitance: 0.01; }
    pin(B) { direction: input;  capacitance: 0.01; }
    pin(Y) {
      direction: output;
      function: "!(A|B)";
      timing() {
        related_pin: "A";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.19"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.13"); }
        fall_transition(scalar){ values("0.04"); }
      }
      timing() {
        related_pin: "B";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.19"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.13"); }
        fall_transition(scalar){ values("0.04"); }
      }
    }
  }

  /* -- 3-input NOR ----------------------------------------------------- */
  cell(NOR3) {
    area : 2.0;
    pin(A) { direction: input;  capacitance: 0.01; }
    pin(B) { direction: input;  capacitance: 0.01; }
    pin(C) { direction: input;  capacitance: 0.01; }
    pin(Y) {
      direction: output;
      function: "!(A|B|C)";
      timing() {
        related_pin: "A";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.24"); }
        rise_transition(scalar){ values("0.06"); }
        cell_fall(scalar)      { values("0.17"); }
        fall_transition(scalar){ values("0.05"); }
      }
      timing() {
        related_pin: "B";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.24"); }
        rise_transition(scalar){ values("0.06"); }
        cell_fall(scalar)      { values("0.17"); }
        fall_transition(scalar){ values("0.05"); }
      }
      timing() {
        related_pin: "C";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.24"); }
        rise_transition(scalar){ values("0.06"); }
        cell_fall(scalar)      { values("0.17"); }
        fall_transition(scalar){ values("0.05"); }
      }
    }
  }

  /* -- 2-input AND ----------------------------------------------------- */
  cell(AND2) {
    area : 2.0;
    pin(A) { direction: input;  capacitance: 0.01; }
    pin(B) { direction: input;  capacitance: 0.01; }
    pin(Y) {
      direction: output;
      function: "(A&B)";
      timing() {
        related_pin: "A";
        timing_sense: positive_unate;
        cell_rise(scalar)      { values("0.24"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.20"); }
        fall_transition(scalar){ values("0.05"); }
      }
      timing() {
        related_pin: "B";
        timing_sense: positive_unate;
        cell_rise(scalar)      { values("0.24"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.20"); }
        fall_transition(scalar){ values("0.05"); }
      }
    }
  }

  /* -- 2-input OR ------------------------------------------------------ */
  cell(OR2) {
    area : 2.0;
    pin(A) { direction: input;  capacitance: 0.01; }
    pin(B) { direction: input;  capacitance: 0.01; }
    pin(Y) {
      direction: output;
      function: "(A|B)";
      timing() {
        related_pin: "A";
        timing_sense: positive_unate;
        cell_rise(scalar)      { values("0.24"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.20"); }
        fall_transition(scalar){ values("0.05"); }
      }
      timing() {
        related_pin: "B";
        timing_sense: positive_unate;
        cell_rise(scalar)      { values("0.24"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.20"); }
        fall_transition(scalar){ values("0.05"); }
      }
    }
  }

  /* -- 2-input XOR ----------------------------------------------------- */
  cell(XOR2) {
    area : 3.5;
    pin(A) { direction: input;  capacitance: 0.01; }
    pin(B) { direction: input;  capacitance: 0.01; }
    pin(Y) {
      direction: output;
      function: "(A^B)";
      timing() {
        related_pin: "A";
        timing_sense: non_unate;
        cell_rise(scalar)      { values("0.35"); }
        rise_transition(scalar){ values("0.07"); }
        cell_fall(scalar)      { values("0.30"); }
        fall_transition(scalar){ values("0.06"); }
      }
      timing() {
        related_pin: "B";
        timing_sense: non_unate;
        cell_rise(scalar)      { values("0.35"); }
        rise_transition(scalar){ values("0.07"); }
        cell_fall(scalar)      { values("0.30"); }
        fall_transition(scalar){ values("0.06"); }
      }
    }
  }

  /* -- 2-input XNOR ---------------------------------------------------- */
  cell(XNOR2) {
    area : 3.5;
    pin(A) { direction: input;  capacitance: 0.01; }
    pin(B) { direction: input;  capacitance: 0.01; }
    pin(Y) {
      direction: output;
      function: "!(A^B)";
      timing() {
        related_pin: "A";
        timing_sense: non_unate;
        cell_rise(scalar)      { values("0.35"); }
        rise_transition(scalar){ values("0.07"); }
        cell_fall(scalar)      { values("0.30"); }
        fall_transition(scalar){ values("0.06"); }
      }
      timing() {
        related_pin: "B";
        timing_sense: non_unate;
        cell_rise(scalar)      { values("0.35"); }
        rise_transition(scalar){ values("0.07"); }
        cell_fall(scalar)      { values("0.30"); }
        fall_transition(scalar){ values("0.06"); }
      }
    }
  }

  /* -- 2-to-1 MUX ------------------------------------------------------ */
  cell(MUX2) {
    area : 3.0;
    pin(A) { direction: input;  capacitance: 0.01; }
    pin(B) { direction: input;  capacitance: 0.01; }
    pin(S) { direction: input;  capacitance: 0.01; }
    pin(Y) {
      direction: output;
      function: "((A & !S) | (B & S))";
      timing() {
        related_pin: "A";
        timing_sense: positive_unate;
        cell_rise(scalar)      { values("0.28"); }
        rise_transition(scalar){ values("0.06"); }
        cell_fall(scalar)      { values("0.24"); }
        fall_transition(scalar){ values("0.05"); }
      }
      timing() {
        related_pin: "B";
        timing_sense: positive_unate;
        cell_rise(scalar)      { values("0.28"); }
        rise_transition(scalar){ values("0.06"); }
        cell_fall(scalar)      { values("0.24"); }
        fall_transition(scalar){ values("0.05"); }
      }
      timing() {
        related_pin: "S";
        timing_sense: non_unate;
        cell_rise(scalar)      { values("0.33"); }
        rise_transition(scalar){ values("0.06"); }
        cell_fall(scalar)      { values("0.27"); }
        fall_transition(scalar){ values("0.05"); }
      }
    }
  }

  /* -- AOI21 ----------------------------------------------------------- */
  cell(AOI21) {
    area : 2.5;
    pin(A) { direction: input;  capacitance: 0.01; }
    pin(B) { direction: input;  capacitance: 0.01; }
    pin(C) { direction: input;  capacitance: 0.01; }
    pin(Y) {
      direction: output;
      function: "!((A&B)|C)";
      timing() {
        related_pin: "A";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.20"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.16"); }
        fall_transition(scalar){ values("0.04"); }
      }
      timing() {
        related_pin: "B";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.20"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.16"); }
        fall_transition(scalar){ values("0.04"); }
      }
      timing() {
        related_pin: "C";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.18"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.14"); }
        fall_transition(scalar){ values("0.04"); }
      }
    }
  }

  /* -- OAI21 ----------------------------------------------------------- */
  cell(OAI21) {
    area : 2.5;
    pin(A) { direction: input;  capacitance: 0.01; }
    pin(B) { direction: input;  capacitance: 0.01; }
    pin(C) { direction: input;  capacitance: 0.01; }
    pin(Y) {
      direction: output;
      function: "!((A|B)&C)";
      timing() {
        related_pin: "A";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.20"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.16"); }
        fall_transition(scalar){ values("0.04"); }
      }
      timing() {
        related_pin: "B";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.20"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.16"); }
        fall_transition(scalar){ values("0.04"); }
      }
      timing() {
        related_pin: "C";
        timing_sense: negative_unate;
        cell_rise(scalar)      { values("0.18"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.14"); }
        fall_transition(scalar){ values("0.04"); }
      }
    }
  }

  /* -- Buffer ---------------------------------------------------------- */
  cell(BUF) {
    area : 1.0;
    pin(A) { direction: input;  capacitance: 0.01; }
    pin(Y) {
      direction: output;
      function: "A";
      timing() {
        related_pin: "A";
        timing_sense: positive_unate;
        cell_rise(scalar)      { values("0.13"); }
        rise_transition(scalar){ values("0.04"); }
        cell_fall(scalar)      { values("0.11"); }
        fall_transition(scalar){ values("0.04"); }
      }
    }
  }

  /* -- D Flip-Flop, positive edge, no reset ----------------------------
   * Maps to Yosys internal cell: $_DFF_P_
   * Setup/hold arcs on D pin omitted for Yosys 0.52 dfflibmap compat.
   */
  cell(DFF) {
    area : 6.0;
    ff(IQ,IQN) {
      clocked_on : "C";
      next_state : "D";
    }
    pin(D) { direction: input;  capacitance: 0.01; }
    pin(C) { direction: input;  capacitance: 0.02; clock: true; }
    pin(Q) {
      direction: output;
      function : "IQ";
      timing() {
        related_pin: "C";
        timing_type: rising_edge;
        cell_rise(scalar)      { values("0.28"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.22"); }
        fall_transition(scalar){ values("0.04"); }
      }
    }
  }

  /* -- D Flip-Flop, positive edge, async active-high clear -------------
   * Maps to Yosys internal cell: $_DFF_PP0_
   * Synchronous-reset DFFs ($SDFF_PP0) are converted via dfflegalize.
   */
  cell(DFFR) {
    area : 8.0;
    ff(IQ,IQN) {
      clocked_on : "C";
      next_state : "D";
      clear      : "R";
    }
    pin(D) { direction: input;  capacitance: 0.01; }
    pin(R) { direction: input;  capacitance: 0.01; }
    pin(C) { direction: input;  capacitance: 0.02; clock: true; }
    pin(Q) {
      direction: output;
      function : "IQ";
      timing() {
        related_pin: "C";
        timing_type: rising_edge;
        cell_rise(scalar)      { values("0.30"); }
        rise_transition(scalar){ values("0.05"); }
        cell_fall(scalar)      { values("0.25"); }
        fall_transition(scalar){ values("0.05"); }
      }
    }
  }
}
'''

with open('/mnt/c/Users/Husai/Ece-410-hardware-ai/codefest/cf07/hdl/generic_130nm.lib', 'w', newline='\n') as f:
    f.write(lib_content)
print('Library written successfully with LF endings')
