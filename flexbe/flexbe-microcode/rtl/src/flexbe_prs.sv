// ============================================================================
// flexbe_prs.sv -- the permute-rotate switch of Sec. 3.1.
//
// The cycle permutation factors as P_f = P_s x P_r: a barrel-shift rotation by
// R composed with one of only M fixed stride permutations selected by S.
// slot j reads bank (pi_S[j] + R) mod P.  Cost is Theta(M*P_bu) muxes.
//
// The stride permutations pi_S are compile-time constants:
//   pi_S[2t]   = ins(t, S)
//   pi_S[2t+1] = pi_S[2t] + 2^S
//
// flexbe_prs      : gather, bank order -> butterfly-slot order (PRS-A)
// flexbe_prs_inv  : scatter, slot order -> bank order          (PRS-B)
// Both expose read_map for verification.
// ============================================================================

module flexbe_prs
  import flexbe_pkg::*;
(
  input  logic [M-1:0]         r,
  input  logic [$clog2(M)-1:0] s,
  input  cplx_t                din  [P],
  output cplx_t                dout [P],
  output logic [M-1:0]         read_map [P]
);
  // compile-time subset permutation table:  perm[s][j] = pi_s[j]
  function automatic logic [M-1:0] pi_s(input int unsigned sel, input int j);
    logic [M-1:0] pi [P];
    for (int t = 0; t < P/2; t++) begin
      logic [M-1:0] ins_lo;
      ins_lo      = (((t >> sel) << (sel+1)) | (t & ((1<<sel)-1)));
      pi[2*t]     = ins_lo;
      pi[2*t+1]   = ins_lo + (1 << sel);
    end
    return pi[j];
  endfunction

  always_comb begin
    for (int j = 0; j < P; j++) begin
      logic [M-1:0] pisj, bank;
      pisj = '0;
      for (int sel = 0; sel < M; sel++)
        if (sel == s) pisj = pi_s(sel, j);
      bank         = pisj + r;         // mod 2^M by wraparound
      read_map[j]  = bank;
      dout[j]      = din[bank];
    end
  end
endmodule : flexbe_prs


module flexbe_prs_inv
  import flexbe_pkg::*;
(
  input  logic [M-1:0]         r,
  input  logic [$clog2(M)-1:0] s,
  input  cplx_t                xin  [P],
  output cplx_t                dout [P]
);
  function automatic logic [M-1:0] pi_s(input int unsigned sel, input int j);
    logic [M-1:0] pi [P];
    for (int t = 0; t < P/2; t++) begin
      logic [M-1:0] ins_lo;
      ins_lo    = (((t >> sel) << (sel+1)) | (t & ((1<<sel)-1)));
      pi[2*t]   = ins_lo;
      pi[2*t+1] = ins_lo + (1 << sel);
    end
    return pi[j];
  endfunction

  // invert the permutation: slot_for_bank(readMap(j)) = j, then dout(b)=xin(slot)
  logic [M-1:0] slot_for_bank [P];
  always_comb begin
    for (int b = 0; b < P; b++) slot_for_bank[b] = '0;
    for (int j = 0; j < P; j++) begin
      logic [M-1:0] pisj, bank;
      pisj = '0;
      for (int sel = 0; sel < M; sel++)
        if (sel == s) pisj = pi_s(sel, j);
      bank                = pisj + r;
      slot_for_bank[bank] = j[M-1:0];
    end
    for (int b = 0; b < P; b++) dout[b] = xin[slot_for_bank[b]];
  end
endmodule : flexbe_prs_inv
