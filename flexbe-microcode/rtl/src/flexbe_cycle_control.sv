// ============================================================================
// flexbe_cycle_control.sv -- per-cycle index vectors, addresses and PRS state.
//
// Combinational realisation of cycle_control(n, P_bu, k, j).  Stage k pairs
// indices differing in the hole bit h = n-1-k; conflict-free access forces two
// regimes (in-row for h<m, two-row for h>=m).  Outputs, for the current (k,j):
//   idx [P]        logical element indices, butterfly-slot order
//   addr[P]        per-bank read/write depth (a_x)
//   r, s           PRS rotation and subset state
//   coeff[P_BU]    coefficient address per butterfly
// ============================================================================
module flexbe_cycle_control
  import flexbe_pkg::*;
  import flexbe_bits::*;
(
  input  logic [$clog2(N_MAX+1)-1:0] n_cur,           // stages (P_sub = 1)
  input  logic [$clog2(N_MAX)-1:0]   k,               // stage index
  input  logic [N_MAX-M:0]           j,               // cycle index in stage
  output logic [IDX_W-1:0]           idx  [P],
  output logic [DEPTH_W-1:0]         addr [P],
  output logic [M-1:0]               r,
  output logic [$clog2(M)-1:0]       s,
  output logic [N_MAX-2:0]           coeff[P_BU]
);

  // hole bit and regime
  logic [$clog2(N_MAX+1)-1:0] h;
  logic in_row;
  assign h      = n_cur - 1 - k;
  assign in_row = (h < M);

  logic       e;
  logic [63:0] j_hi;
  assign e    = j[0];
  assign j_hi = j >> 1;

  // lower indices I[2t]
  logic [IDX_W-1:0] lower [P_BU];
  always_comb begin
    for (int t = 0; t < P_BU; t++) begin
      logic [63:0] ins_t, h_in;
      logic [63:0] h_two, htwo_row;
      int unsigned hmm;
      // in-row: H = j; I[2t] = (H<<m) + ins(t,h)
      ins_t = insert_zero(t, h, M);
      h_in  = (({{(64-($bits(j))){1'b0}}, j}) << M) | ins_t;
      // two-row: H = ins(j>>1, h-m); I[2t] = (H<<m) + 2t + e
      hmm      = in_row ? 0 : (h - M);
      htwo_row = insert_zero(j_hi, hmm, N_MAX - M + 1);
      h_two    = (htwo_row << M) | (2*t) | e;
      lower[t] = in_row ? h_in[IDX_W-1:0] : h_two[IDX_W-1:0];
    end
  end

  // full index vector: interleave lower / upper with stride 2^h
  logic [63:0] two_h;
  assign two_h = 64'd1 << h;
  always_comb begin
    for (int t = 0; t < P_BU; t++) begin
      idx[2*t]   = lower[t];
      idx[2*t+1] = (lower[t] + two_h[IDX_W-1:0]);
    end
  end

  // addresses: addr(bsm(idx)) = idx >> m
  always_comb begin
    for (int b = 0; b < P; b++) addr[b] = '0;
    for (int ss = 0; ss < P; ss++) begin
      logic [M-1:0] bank;
      bank = bsm(idx[ss], M);
      addr[bank] = idx[ss] >> M;
    end
  end

  // PRS control
  assign r = bsm(idx[0], M);
  always_comb begin
    if (in_row) s = h[$clog2(M)-1:0];
    else        s = '0;
  end

  // coefficient address per butterfly:
  //   a = I[2t]; coeff = ((a >> (h+1)) << h) | (a & (2^h - 1))
  always_comb begin
    for (int t = 0; t < P_BU; t++) begin
      logic [63:0] a, hi, lo_m, cf;
      a    = lower[t];
      hi   = a >> (h + 1);
      lo_m = (64'd1 << h) - 64'd1;
      cf   = (hi << h) | (a & lo_m);
      coeff[t] = cf[N_MAX-2:0];
    end
  end

endmodule : flexbe_cycle_control
