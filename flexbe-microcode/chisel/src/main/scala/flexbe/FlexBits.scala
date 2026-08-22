// =============================================================================
// FlexBits.scala -- combinational bit primitives shared by the datapath and control.
//
// These mirror flexbe.py exactly (popcount, bit_rev, insert_zero, bsm), so the
// Chisel elaboration and the Python cycle-level model agree bit for bit.  The
// definitions here are pure combinational helpers; they instantiate no state.
// =============================================================================
package flexbe

import chisel3._
import chisel3.util._

object FlexBits {

  /** Population count of an arbitrary-width UInt. */
  def popcount(x: UInt): UInt = PopCount(x.asBools)

  /** Reverse the low `n` bits of `x` (bit_rev_n of the paper). */
  def bitRev(x: UInt, n: Int): UInt = {
    require(n >= 1)
    Cat((0 until n).map(i => x(i)))            // MSB..LSB of the reversal
  }

  /** ins(x, p): insert a 0 bit at position p, widening by one bit.
    *   result = ((x >> p) << (p+1)) | (x & (2^p - 1))
    * `p` is a hardware value in [0, maxP]; a small mux tree covers it.
    */
  def insertZero(x: UInt, p: UInt, maxP: Int, outW: Int): UInt = {
    val cases = (0 to maxP).map { pp =>
      val lo = if (pp == 0) 0.U else x(pp - 1, 0)
      val hi = (x >> pp.U).asUInt
      pp.U -> ((hi << (pp + 1).U).asUInt | lo).apply(outW - 1, 0)
    }
    MuxLookup(p, 0.U)(cases)
  }

  /** bsm(I, m) = (popcount(I >> m) + (I mod 2^m)) mod 2^m  -- Eq. (3).
    * A RAM row is 2^m consecutive indices; row H is rotated by popcount(H)
    * banks, which is the whole basis of the conflict-free schedule.
    */
  def bsm(I: UInt, m: Int): UInt = {
    val mask = (1 << m) - 1
    val low  = I(m - 1, 0)
    val high = (I >> m.U).asUInt
    (popcount(high) + low)(m - 1, 0)           // mod 2^m by truncation
  }
}
