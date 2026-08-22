// ============================================================================
// flexbe_descriptor.sv -- decode a 32-byte command word.
//
// Byte layout, little-endian, matching Descriptor.encode() in microcode.py
// (struct '<8B6I'): word0 bits [31:0] hold opcode(0), flags(1), mode(2),
// log2_l(3); word1 holds log2_Psub(4), stage_lo(5), stage_hi(6), pool(7);
// words 2..7 are src, dst, coeff, count, P_N, tag.
// ============================================================================
package flexbe_desc;
  import flexbe_pkg::*;

  typedef struct packed {
    logic [7:0]  opcode;
    logic [7:0]  flags;
    logic [7:0]  mode;
    logic [7:0]  log2_l;
    logic [7:0]  log2_psub;
    logic [7:0]  stage_lo;
    logic [7:0]  stage_hi;
    logic [7:0]  pool;
    logic [31:0] src;
    logic [31:0] dst;
    logic [31:0] coeff;
    logic [31:0] count;
    logic [31:0] p_n;
    logic [31:0] tag;
  } descriptor_t;

  function automatic descriptor_t decode(input logic [255:0] raw);
    descriptor_t d;
    d.opcode    = raw[7:0];
    d.flags     = raw[15:8];
    d.mode      = raw[23:16];
    d.log2_l    = raw[31:24];
    d.log2_psub = raw[39:32];
    d.stage_lo  = raw[47:40];
    d.stage_hi  = raw[55:48];
    d.pool      = raw[63:56];
    d.src       = raw[95:64];
    d.dst       = raw[127:96];
    d.coeff     = raw[159:128];
    d.count     = raw[191:160];
    d.p_n       = raw[223:192];
    d.tag       = raw[255:224];
    return d;
  endfunction

  function automatic logic [1:0] res_class(input logic [7:0] opcode);
    case (opcode)
      OP_LOAD, OP_WLOAD : return RC_DMA_IN;
      OP_STORE          : return RC_DMA_OUT;
      default           : return RC_COMPUTE;
    endcase
  endfunction

endpackage : flexbe_desc
