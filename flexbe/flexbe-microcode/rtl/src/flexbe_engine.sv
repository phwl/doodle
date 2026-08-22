// ============================================================================
// flexbe_engine.sv -- one butterfly engine: the per-cycle loop of Fig. 2.
//
// Each cycle:
//   1. cycle_control emits idx / addr / r / s / coeff for (k, j)
//   2. read one word per bank from BSPM at addr
//   3. PRS-A gathers bank order -> butterfly-slot order
//   4. P_BU butterfly units apply 2x2 coefficients from CRAM
//   5. PRS-B scatters slot order -> bank order
//   6. write back in place to BSPM at addr
//
// An FSM walks (k, j) over stages*(N/P) cycles for one bs.bfly command.  BSPM
// is a synchronous-read memory (one-cycle latency); read data of cycle t is
// used with control that is registered one cycle to align with it.
// ============================================================================
module flexbe_engine
  import flexbe_pkg::*;
#(
  parameter int BSPM_DEPTH   = 8192,
  parameter int CRAM_DEPTH   = 4096
)(
  input  logic clk,
  input  logic rst_n,

  input  logic                       start,
  input  logic [$clog2(N_MAX+1)-1:0] n_cur,     // stages for this command
  input  logic                       scale,     // FFT per-stage scaling
  input  logic [$clog2(CRAM_DEPTH)-1:0] coeff_base,

  output logic busy,
  output logic done,

  // verification hooks
  output logic [IDX_W-1:0] dbg_idx [P],
  output logic [M-1:0]     dbg_r,
  output logic [$clog2(M)-1:0] dbg_s
);
  localparam int BAW = $clog2(BSPM_DEPTH);
  localparam int CAW = $clog2(CRAM_DEPTH);

  // ---- control FSM: iterate (k, j) ----------------------------------------
  typedef enum logic [1:0] {S_IDLE, S_RUN, S_DRAIN} state_t;
  state_t state, state_n;

  logic [$clog2(N_MAX)-1:0] k;
  logic [N_MAX-M:0]         j;
  logic [N_MAX-M:0]         cyc_per_stage;   // N/P for P_sub = 1
  logic [N_MAX-M:0]         last_j;
  logic [$clog2(N_MAX)-1:0] last_k;

  assign cyc_per_stage = (1 << (n_cur - M));
  assign last_j        = cyc_per_stage - 1;
  assign last_k        = n_cur - 1;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE; k <= '0; j <= '0;
    end else begin
      case (state)
        S_IDLE:  if (start) begin state <= S_RUN; k <= '0; j <= '0; end
        S_RUN: begin
          if (j == last_j) begin
            j <= '0;
            if (k == last_k) state <= S_DRAIN;
            else             k <= k + 1'b1;
          end else j <= j + 1'b1;
        end
        S_DRAIN: state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end

  assign busy = (state != S_IDLE);
  assign done = (state == S_DRAIN);

  // ---- cycle control ------------------------------------------------------
  logic [IDX_W-1:0]     idx  [P];
  logic [DEPTH_W-1:0]   addr [P];
  logic [M-1:0]         r_ctl;
  logic [$clog2(M)-1:0] s_ctl;
  logic [N_MAX-2:0]     coeff[P_BU];

  flexbe_cycle_control u_ctl (
    .n_cur(n_cur), .k(k), .j(j),
    .idx(idx), .addr(addr), .r(r_ctl), .s(s_ctl), .coeff(coeff)
  );

  // ---- BSPM ---------------------------------------------------------------
  logic [BAW-1:0] r_addr [P], w_addr [P];
  cplx_t          r_data [P], w_data [P];
  logic           w_en;

  // register control one cycle to align with synchronous read data
  logic [M-1:0]         r_q;
  logic [$clog2(M)-1:0] s_q;
  logic [DEPTH_W-1:0]   addr_q [P];
  logic                 run_q;
  always_ff @(posedge clk) begin
    r_q   <= r_ctl;
    s_q   <= s_ctl;
    for (int b = 0; b < P; b++) addr_q[b] <= addr[b];
    run_q <= (state == S_RUN);
  end

  always_comb for (int b = 0; b < P; b++) r_addr[b] = addr[b];

  flexbe_banked_mem #(.DEPTH(BSPM_DEPTH)) u_bspm (
    .clk(clk),
    .r_addr(r_addr), .r_data(r_data),
    .w_en(w_en), .w_addr(w_addr), .w_data(w_data)
  );

  // ---- PRS-A gather -------------------------------------------------------
  cplx_t gathered [P];
  logic [M-1:0] rmap [P];
  flexbe_prs u_prs_a (
    .r(r_q), .s(s_q), .din(r_data), .dout(gathered), .read_map(rmap)
  );

  // ---- CRAM + butterflies -------------------------------------------------
  // CRAM entry = packed {c00, c01, c10, c11}
  typedef struct packed { cplx_t c00, c01, c10, c11; } coeff_t;
  coeff_t cram [CRAM_DEPTH];
  coeff_t cram_rd [P_BU];
  logic [N_MAX-2:0] coeff_q [P_BU];
  always_ff @(posedge clk)
    for (int t = 0; t < P_BU; t++) begin
      coeff_q[t] <= coeff[t];
      cram_rd[t] <= cram[coeff[t] + coeff_base];
    end

  cplx_t bo0 [P_BU], bo1 [P_BU];
  genvar t;
  generate
    for (t = 0; t < P_BU; t++) begin : g_bu
      flexbe_butterfly u_bu (
        .a(gathered[2*t]), .b(gathered[2*t+1]),
        .c00(cram_rd[t].c00), .c01(cram_rd[t].c01),
        .c10(cram_rd[t].c10), .c11(cram_rd[t].c11),
        .scale(scale),
        .out0(bo0[t]), .out1(bo1[t])
      );
    end
  endgenerate

  // ---- PRS-B scatter ------------------------------------------------------
  cplx_t bx [P];
  always_comb
    for (int tt = 0; tt < P_BU; tt++) begin
      bx[2*tt]   = bo0[tt];
      bx[2*tt+1] = bo1[tt];
    end
  cplx_t scattered [P];
  flexbe_prs_inv u_prs_b (
    .r(r_q), .s(s_q), .xin(bx), .dout(scattered)
  );

  // ---- write back in place ------------------------------------------------
  assign w_en = run_q;
  always_comb for (int b = 0; b < P; b++) begin
    w_addr[b] = addr_q[b];
    w_data[b] = scattered[b];
  end

  assign dbg_idx = idx;
  assign dbg_r   = r_ctl;
  assign dbg_s   = s_ctl;

endmodule : flexbe_engine
