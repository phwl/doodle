// ============================================================================
// flexbe_banked_mem.sv -- the P-bank butterfly scratchpad (BSPM).
//
// P single-port banks, each DEPTH words.  Each cycle one word is read and one
// written per bank, each bank at its own address (the two-row regime reads two
// depths at once, so a single shared address will not do).  Synchronous read,
// one-cycle latency.
// ============================================================================
module flexbe_banked_mem
  import flexbe_pkg::*;
#(
  parameter int DEPTH = 8192
)(
  input  logic                     clk,
  input  logic [$clog2(DEPTH)-1:0] r_addr [P],
  output cplx_t                    r_data [P],
  input  logic                     w_en,
  input  logic [$clog2(DEPTH)-1:0] w_addr [P],
  input  cplx_t                    w_data [P]
);
  localparam int AW = $clog2(DEPTH);

  genvar b;
  generate
    for (b = 0; b < P; b++) begin : g_bank
      cplx_t mem [DEPTH];
      logic [AW-1:0] r_addr_q;

      always_ff @(posedge clk) begin
        if (w_en) mem[w_addr[b]] <= w_data[b];
        r_addr_q <= r_addr[b];
      end
      assign r_data[b] = mem[r_addr_q];
    end
  endgenerate
endmodule : flexbe_banked_mem
