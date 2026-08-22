// ============================================================================
// flexbe_bits.sv -- combinational bit primitives (popcount, bit_rev,
// insert_zero, bsm), matching flexbe.py exactly.  Pure functions in a package
// so every module shares one definition.
// ============================================================================
package flexbe_bits;

  // population count of a W-bit value
  function automatic int unsigned popcount(input logic [63:0] x, input int W);
    int unsigned c;
    c = 0;
    for (int i = 0; i < W; i++) c += x[i];
    return c;
  endfunction

  // reverse the low n bits of x
  function automatic logic [63:0] bit_rev(input logic [63:0] x, input int n);
    logic [63:0] r;
    r = '0;
    for (int i = 0; i < n; i++) r[n-1-i] = x[i];
    return r;
  endfunction

  // ins(x, p): insert a 0 bit at position p, result truncated to out_w bits
  //   ((x >> p) << (p+1)) | (x & (2^p - 1))
  function automatic logic [63:0] insert_zero(input logic [63:0] x,
                                              input int unsigned p,
                                              input int out_w);
    logic [63:0] lo, hi, res;
    lo  = x & ((64'd1 << p) - 64'd1);
    hi  = x >> p;
    res = (hi << (p + 1)) | lo;
    return res & ((64'd1 << out_w) - 64'd1);
  endfunction

  // bsm(I, m) = (popcount(I >> m) + (I mod 2^m)) mod 2^m   -- Eq. (3)
  function automatic logic [7:0] bsm(input logic [63:0] I, input int m);
    logic [63:0] low, high;
    int unsigned s;
    low  = I & ((64'd1 << m) - 64'd1);
    high = I >> m;
    s    = popcount(high, 64) + low[31:0];
    return s[m-1:0];
  endfunction

endpackage : flexbe_bits
