// ============================================================================
// flexbe_pkg.sv -- shared parameters, constants and the fixed-point complex type
//
// Mirrors flexbe.py / microcode.py so a descriptor stream and coefficient set
// produced by the Python model load directly into this RTL.
// ============================================================================
package flexbe_pkg;

  // ---- datapath geometry --------------------------------------------------
  localparam int P_BU   = 16;             // butterfly units
  localparam int P      = 2 * P_BU;       // switching ports / RAM banks = 32
  localparam int M      = $clog2(P);      // 5
  localparam int N_MAX  = 15;             // largest transform: 2^15 = 32768
  localparam int IDX_W  = N_MAX;          // logical element index width
  localparam int DEPTH_W= N_MAX - M;      // per-bank depth address width (a_x)

  // ---- fixed point (Q1.15, 32-bit accumulate) -----------------------------
  localparam int INT_BITS  = 1;
  localparam int FRAC_BITS = 15;
  localparam int WORD_W    = INT_BITS + FRAC_BITS;   // 16
  localparam int ACC_W     = 2 * WORD_W + 2;         // product + carries

  typedef struct packed {
    logic signed [WORD_W-1:0] re;
    logic signed [WORD_W-1:0] im;
  } cplx_t;

  // ---- opcodes (== microcode.py OPCODES) ----------------------------------
  localparam logic [7:0] OP_LOAD  = 8'h01;
  localparam logic [7:0] OP_WLOAD = 8'h02;
  localparam logic [7:0] OP_STORE = 8'h03;
  localparam logic [7:0] OP_BFLY  = 8'h10;
  localparam logic [7:0] OP_BREV  = 8'h11;
  localparam logic [7:0] OP_POW   = 8'h20;
  localparam logic [7:0] OP_MAG   = 8'h21;
  localparam logic [7:0] OP_POOL  = 8'h22;

  // ---- modes --------------------------------------------------------------
  localparam logic [7:0] MODE_FFT       = 8'd0;
  localparam logic [7:0] MODE_IFFT      = 8'd1;
  localparam logic [7:0] MODE_BL        = 8'd2;
  localparam logic [7:0] MODE_BL_EXPAND = 8'd3;

  // ---- flags (bit masks, == microcode.py FLAGS) ---------------------------
  localparam logic [7:0] FLAG_BITREV    = 8'h01;
  localparam logic [7:0] FLAG_MAG       = 8'h02;
  localparam logic [7:0] FLAG_NORMALISE = 8'h04;
  localparam logic [7:0] FLAG_RELU      = 8'h08;
  localparam logic [7:0] FLAG_LAYERNORM = 8'h10;
  localparam logic [7:0] FLAG_SHORTCUT  = 8'h20;
  localparam logic [7:0] FLAG_MAXPOOL   = 8'h40;

  // ---- resource classes ---------------------------------------------------
  localparam logic [1:0] RC_COMPUTE = 2'd0;
  localparam logic [1:0] RC_DMA_IN  = 2'd1;
  localparam logic [1:0] RC_DMA_OUT = 2'd2;

endpackage : flexbe_pkg
