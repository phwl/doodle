// =============================================================================
// Complex.scala -- Q1.15 complex arithmetic and the radix-2 butterfly unit.
//
// Matches FixedPointFormat in flexbe.py: 16-bit words, 15 fractional bits,
// 32-bit accumulation, round-to-nearest-even, saturation.  The butterfly
// applies a general 2x2 coefficient matrix, so one unit serves both the FFT
// (coefficients [[1,1],[W,-W]]) and a trained butterfly-linear layer.
// =============================================================================
package flexbe

import chisel3._
import chisel3.util._

/** A fixed-point complex sample in Q(intBits).(fracBits). */
class Cplx(val intBits: Int = 1, val fracBits: Int = 15) extends Bundle {
  val w  = intBits + fracBits
  val re = SInt((intBits + fracBits).W)
  val im = SInt((intBits + fracBits).W)
}

object Cplx {
  def apply(intBits: Int = 1, fracBits: Int = 15): Cplx = new Cplx(intBits, fracBits)

  def wire(re: SInt, im: SInt, intBits: Int = 1, fracBits: Int = 15): Cplx = {
    val c = Wire(new Cplx(intBits, fracBits)); c.re := re; c.im := im; c
  }
}

/** Fixed-point helpers on the wide accumulator. */
object Fx {
  /** Round-to-nearest-even then saturate a `frac+guard`-bit accumulator back to
    * a Q(intBits).(fracBits) word.  `acc` carries `fracBits` extra low bits
    * from a full-precision product. */
  def roundSat(acc: SInt, intBits: Int, fracBits: Int): SInt = {
    val w      = intBits + fracBits
    val shift  = fracBits
    // convergent rounding: add half LSB, then correct the exactly-.5 tie to even
    val half   = (BigInt(1) << (shift - 1)).S
    val biased = acc +& half
    val shifted = (biased >> shift).asSInt
    // tie-to-even: if the discarded bits were exactly 0.5, clear the LSB
    val stickyMask = ((BigInt(1) << (shift - 1)) - 1)
    val exactlyHalf = (acc(shift - 1) === 1.U) && ((acc.asUInt & stickyMask.U) === 0.U)
    val rounded = Mux(exactlyHalf, (shifted & ~1.S(shifted.getWidth.W)).asSInt, shifted)
    val hi = ((BigInt(1) << (w - 1)) - 1).S
    val lo = (-(BigInt(1) << (w - 1))).S
    Mux(rounded > hi, hi, Mux(rounded < lo, lo, rounded))(w - 1, 0).asSInt
  }
}

/** One radix-2 butterfly applying a general 2x2 complex matrix:
  *   out0 = c00*a + c01*b
  *   out1 = c10*a + c11*b
  * FFT twiddles set c00=c01=1, c10=W, c11=-W; BL layers supply trained weights.
  */
class ButterflyUnit(intBits: Int = 1, fracBits: Int = 15) extends Module {
  val io = IO(new Bundle {
    val a    = Input(new Cplx(intBits, fracBits))
    val b    = Input(new Cplx(intBits, fracBits))
    val c00  = Input(new Cplx(intBits, fracBits))
    val c01  = Input(new Cplx(intBits, fracBits))
    val c10  = Input(new Cplx(intBits, fracBits))
    val c11  = Input(new Cplx(intBits, fracBits))
    val scale = Input(Bool())                    // halve outputs (FFT stage scaling)
    val out0 = Output(new Cplx(intBits, fracBits))
    val out1 = Output(new Cplx(intBits, fracBits))
  })

  // full-precision complex multiply-accumulate into a wide accumulator
  def cmul(x: Cplx, y: Cplx): (SInt, SInt) = {
    val rr = x.re * y.re
    val ii = x.im * y.im
    val ri = x.re * y.im
    val ir = x.im * y.re
    ((rr -& ii), (ri +& ir))
  }
  def cadd(p: (SInt, SInt), q: (SInt, SInt)): (SInt, SInt) =
    ((p._1 +& q._1), (p._2 +& q._2))

  val o0 = cadd(cmul(io.c00, io.a), cmul(io.c01, io.b))
  val o1 = cadd(cmul(io.c10, io.a), cmul(io.c11, io.b))

  def finish(acc: (SInt, SInt)): Cplx = {
    val re = Mux(io.scale, (acc._1 >> 1).asSInt, acc._1)
    val im = Mux(io.scale, (acc._2 >> 1).asSInt, acc._2)
    Cplx.wire(Fx.roundSat(re, intBits, fracBits),
              Fx.roundSat(im, intBits, fracBits), intBits, fracBits)
  }

  io.out0 := finish(o0)
  io.out1 := finish(o1)
}
