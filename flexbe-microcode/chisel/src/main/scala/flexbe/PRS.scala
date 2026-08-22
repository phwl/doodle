// =============================================================================
// PRS.scala -- the permute-rotate switch of Sec. 3.1.
//
// The full permutation each cycle factors as P_f = P_s x P_r: a barrel-shift
// rotation by R composed with one of only m fixed stride permutations selected
// by S.  slot j reads bank (pi_S[j] + R) mod P.  Cost is Theta(m*P_bu) 2:1
// muxes, versus Theta(P_bu^2) for a general crossbar.
//
// Parameterised over P = 2*P_bu (a power of two, m = log2 P).  The stride
// permutations pi_S are compile-time constants:
//   pi_S[2t]   = ins(t, S)          (insert a zero bit at position S)
//   pi_S[2t+1] = pi_S[2t] + 2^S
// =============================================================================
package flexbe

import chisel3._
import chisel3.util._

/** Compile-time table of the m subset permutations for a given P. */
object SubsetPerm {
  def table(P: Int): Seq[Seq[Int]] = {
    val m = log2Ceil(P)
    (0 until m).map { s =>
      val pi = Array.fill(P)(0)
      for (t <- 0 until P / 2) {
        val insLo = ((t >> s) << (s + 1)) | (t & ((1 << s) - 1)) // ins(t, s)
        pi(2 * t) = insLo
        pi(2 * t + 1) = insLo + (1 << s)
      }
      pi.toSeq
    }
  }
}

class PRS(val P: Int) extends Module {
  require((P & (P - 1)) == 0 && P >= 2, "P must be a power of two")
  val m = log2Ceil(P)

  val io = IO(new Bundle {
    val R    = Input(UInt(m.W))
    val S    = Input(UInt((if (m > 1) log2Ceil(m) else 1).W))
    /** PRS-A gather: bank-ordered in -> butterfly-slot-ordered out. */
    val din  = Input(Vec(P, new Cplx))
    val dout = Output(Vec(P, new Cplx))
    /** the read map itself, exposed for verification: dout(j) = din(readMap(j)) */
    val readMap = Output(Vec(P, UInt(m.W)))
  })

  val perms = SubsetPerm.table(P)              // perms(S)(j) = pi_S[j]

  // pi_S[j] selected by S, then rotated by R:  bank = (pi_S[j] + R) mod P
  for (j <- 0 until P) {
    val piSj =
      if (m == 1) 0.U(m.W)                      // P=2: only identity
      else MuxLookup(io.S, 0.U(m.W))(
        (0 until m).map(s => s.U -> perms(s)(j).U(m.W)))
    val bank = (piSj +% io.R)(m - 1, 0)        // +% wraps mod 2^m
    io.readMap(j) := bank
    io.dout(j) := io.din(bank)
  }
}

/** PRS-B scatter: the inverse of PRS.  Writes X back to bank order:
  *   dbank(readMap(j)) = xin(j)
  */
class PRSInverse(val P: Int) extends Module {
  require((P & (P - 1)) == 0 && P >= 2)
  val m = log2Ceil(P)
  val io = IO(new Bundle {
    val R    = Input(UInt(m.W))
    val S    = Input(UInt((if (m > 1) log2Ceil(m) else 1).W))
    val xin  = Input(Vec(P, new Cplx))
    val dout = Output(Vec(P, new Cplx))
  })
  val perms = SubsetPerm.table(P)

  // Forward read map, identical to PRS: bankOf(j) = (pi_S[j] + R) mod P.
  val bankOf = Wire(Vec(P, UInt(m.W)))
  for (j <- 0 until P) {
    val piSj =
      if (m == 1) 0.U(m.W)
      else MuxLookup(io.S, 0.U(m.W))(
        (0 until m).map(s => s.U -> perms(s)(j).U(m.W)))
    bankOf(j) := (piSj +% io.R)(m - 1, 0)
  }

  // Invert combinationally: for each destination bank b, select the slot j whose
  // bankOf(j) == b.  Written as an explicit per-bank reduction so every element
  // of dout is unconditionally driven (no runtime-indexed sink).
  for (b <- 0 until P) {
    val hit = VecInit((0 until P).map(j => bankOf(j) === b.U))
    val sel = VecInit((0 until P).map(j => io.xin(j)))
    io.dout(b) := Mux1H(hit, sel)          // readMap is a permutation: exactly one hit
  }
}
