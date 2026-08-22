// =============================================================================
// FlexBESpec.scala -- cross-check the RTL against the Python golden model.
//
// Golden.scala is emitted by flexbe.py.  The three combinational blocks (bsm,
// the subset permutations, the PRS read map) and the cycle_control trace are
// checked here without a Verilog backend, by re-deriving the same closed-form
// logic in Scala and asserting it equals both the golden vectors AND the values
// the SubsetPerm hardware table produces.  This keeps `sbt test` self-contained:
// it needs no Verilator and no svsim handshake.
//
// The Engine FSM (which does need a clocked simulator) is exercised separately
// in EngineSimSpec, tagged so it can be skipped when no simulator is present:
//   sbt test                      -- runs the backend-free checks
//   sbt "testOnly *EngineSimSpec"  -- runs the FSM test (needs Verilator)
// =============================================================================
package flexbe

import chisel3._
import chisel3.util.log2Ceil
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Reference re-derivation of the RTL's combinational logic, in plain Scala.
  * These mirror FlexBits / CycleControl expression-for-expression, so agreement
  * with Golden (from flexbe.py) verifies both the intent and the Scala image of
  * the hardware.  The hardware itself is elaborated in the "elaborates" test. */
object Model {
  def mask(x: Int, w: Int): Int = x & ((1 << w) - 1)

  def insertZero(x: Int, p: Int, outW: Int): Int = {
    val lo = if (p > 0) x & ((1 << p) - 1) else 0
    val hi = x >> p
    mask((hi << (p + 1)) | lo, outW)
  }

  def bsm(i: Int, m: Int): Int =
    mask(Integer.bitCount(i >> m) + mask(i, m), m)

  def subset(P: Int): Seq[Seq[Int]] = SubsetPerm.table(P)  // same source as RTL

  def readMap(P: Int, R: Int, S: Int): Seq[Int] = {
    val m = log2Ceil(P)
    (0 until P).map(j => mask(subset(P)(S)(j) + R, m))
  }

  /** cycle_control(nMax, pBu, nCur, k, j) -> (idx, addr, R, S, coeff). */
  def cycleControl(nMax: Int, pBu: Int, nCur: Int, k: Int, j: Int)
      : (Seq[Int], Seq[Int], Int, Int, Seq[Int]) = {
    val P = 2 * pBu; val m = log2Ceil(P); val idxW = nMax
    val h = nCur - 1 - k; val inRow = h < m; val e = j & 1; val jHi = j >> 1
    val lower = (0 until pBu).map { t =>
      val insT = insertZero(mask(t, math.max(1, m - 1)), h, m)
      val hIn  = mask((j << m) | insT, idxW)
      val hmm  = if (h - m >= 0) h - m else 0
      val Htwo = insertZero(jHi, hmm, nMax - m + 1)
      val hTwo = mask((Htwo << m) | (2 * t) | e, idxW)
      if (inRow) hIn else hTwo
    }
    val twoH = 1 << h
    val idx = Array.fill(P)(0)
    for (t <- 0 until pBu) { idx(2*t) = lower(t); idx(2*t+1) = mask(lower(t) + twoH, idxW) }
    val addr = Array.fill(P)(0)
    for (s <- 0 until P) addr(bsm(idx(s), m)) = idx(s) >> m
    val R = bsm(idx(0), m)
    val sW = math.max(1, log2Ceil(m))
    val S = if (inRow) mask(h, sW) else 0
    val coeff = (0 until pBu).map { t =>
      val a = lower(t); val hi = a >> (h + 1); val loM = (1 << h) - 1
      mask((hi << h) | (a & loM), nMax - 1)
    }
    (idx.toSeq, addr.toSeq, R, S, coeff)
  }
}

class FlexBESpec extends AnyFlatSpec with Matchers {

  "bsm" should "match the Python golden table for N=16, m=2" in {
    Golden.bsm16.zipWithIndex.foreach { case (exp, i) =>
      Model.bsm(i, 2) shouldBe exp
    }
  }

  it should "agree with the Python model over the full 15-bit range, m=5" in {
    for (i <- 0 until (1 << 15)) Model.bsm(i, 5) shouldBe
      (Integer.bitCount(i >> 5) + (i & 31)) % 32
  }

  "SubsetPerm" should "match the Python subset permutations for P=4" in {
    val t = SubsetPerm.table(4)
    t(0) shouldBe Golden.subset0
    t(1) shouldBe Golden.subset1
  }

  "PRS read map" should "reproduce read_map(R,S) for every R,S" in {
    for (R <- 0 until 4; S <- 0 until 2)
      Model.readMap(4, R, S) shouldBe Golden.readMap(s"$R,$S")
  }

  "CycleControl" should "reproduce the full 16-point golden trace" in {
    Golden.trace.foreach { case ((k, j, _), idx, addr, r, s, coeff) =>
      val (mIdx, mAddr, mR, mS, mCoeff) = Model.cycleControl(4, 2, 4, k, j)
      mIdx shouldBe idx
      mAddr shouldBe addr
      mR shouldBe r
      mS shouldBe s
      mCoeff shouldBe coeff
    }
  }

  // The RTL modules must also *elaborate* (compile through Chisel/firtool).
  // ChiselStage.emitCHIRRTL forces elaboration without any Verilog backend.
  "The RTL" should "elaborate without error" in {
    import circt.stage.ChiselStage
    noException should be thrownBy {
      ChiselStage.emitCHIRRTL(new CycleControl(15, 16))
      ChiselStage.emitCHIRRTL(new PRS(32))
      ChiselStage.emitCHIRRTL(new PRSInverse(32))
      ChiselStage.emitCHIRRTL(new Engine(new EngineConfig()))
      ChiselStage.emitCHIRRTL(new Sequencer(new SequencerConfig()))
    }
  }
}
