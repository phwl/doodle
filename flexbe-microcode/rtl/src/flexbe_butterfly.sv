// ============================================================================
// flexbe_butterfly.sv -- one radix-2 butterfly applying a general 2x2 matrix.
//
//   out0 = c00*a + c01*b
//   out1 = c10*a + c11*b
//
// Full-precision complex products accumulate in a wide accumulator, which is
// then round-to-nearest-even'd and saturated back to Q1.15 (Sec. 5.2).  FFT
// twiddles set c00=c01=1, c10=W, c11=-W; BL layers supply trained weights.
// ============================================================================
module flexbe_butterfly
  import flexbe_pkg::*;
(
  input  cplx_t a,
  input  cplx_t b,
  input  cplx_t c00,
  input  cplx_t c01,
  input  cplx_t c10,
  input  cplx_t c11,
  input  logic  scale,          // halve outputs (FFT per-stage scaling)
  output cplx_t out0,
  output cplx_t out1
);

  // complex multiply -> (re, im) at full precision
  function automatic void cmul(input cplx_t x, input cplx_t y,
                               output logic signed [ACC_W-1:0] pr,
                               output logic signed [ACC_W-1:0] pi);
    pr = (x.re * y.re) - (x.im * y.im);
    pi = (x.re * y.im) + (x.im * y.re);
  endfunction

  // round-to-nearest-even then saturate a Q1.(2*FRAC) accumulator to Q1.15
  function automatic logic signed [WORD_W-1:0] round_sat(
      input logic signed [ACC_W-1:0] acc);
    logic signed [ACC_W-1:0] biased, shifted;
    logic exact_half;
    logic signed [WORD_W-1:0] hi, lo;
    biased  = acc + (1 <<< (FRAC_BITS-1));
    shifted = biased >>> FRAC_BITS;
    // tie to even: discarded bits exactly 0.5 -> clear LSB
    exact_half = acc[FRAC_BITS-1] &&
                 ((acc & ((1 <<< (FRAC_BITS-1)) - 1)) == '0);
    if (exact_half) shifted[0] = 1'b0;
    hi = (1 <<< (WORD_W-1)) - 1;
    lo = -(1 <<< (WORD_W-1));
    if (shifted > hi)      return hi;
    else if (shifted < lo) return lo;
    else                   return shifted[WORD_W-1:0];
  endfunction

  logic signed [ACC_W-1:0] a00r, a00i, a01r, a01i, o0r, o0i;
  logic signed [ACC_W-1:0] a10r, a10i, a11r, a11i, o1r, o1i;

  always_comb begin
    cmul(c00, a, a00r, a00i);
    cmul(c01, b, a01r, a01i);
    cmul(c10, a, a10r, a10i);
    cmul(c11, b, a11r, a11i);
    o0r = a00r + a01r;  o0i = a00i + a01i;
    o1r = a10r + a11r;  o1i = a10i + a11i;
    if (scale) begin
      o0r = o0r >>> 1;  o0i = o0i >>> 1;
      o1r = o1r >>> 1;  o1i = o1i >>> 1;
    end
    out0.re = round_sat(o0r);  out0.im = round_sat(o0i);
    out1.re = round_sat(o1r);  out1.im = round_sat(o1i);
  end

endmodule : flexbe_butterfly
