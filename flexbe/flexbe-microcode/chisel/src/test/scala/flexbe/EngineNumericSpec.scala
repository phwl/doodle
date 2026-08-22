// =============================================================================
// EngineNumericSpec.scala -- end-to-end numeric check of the datapath.
//
// Loads a 16-point input and the per-stage FFT twiddles into the Engine through
// its debug ports, runs one transform (Q1.15, stage scaling on, no write-back),
// reads the bit-reversed result back, and compares against NumGolden -- the same
// values the fixed-point Python model produces.  This is the test that verifies
// the butterfly arithmetic in hardware, not just the control sequencing.
//
// Needs a Verilog backend; self-cancels if the svsim handshake is unavailable.
// =============================================================================
package flexbe

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EngineNumericSpec extends AnyFlatSpec with Matchers {

  "Engine" should "compute a 16-point FFT matching the fixed-point model" in {
    val cfg = new EngineConfig(nMax = 4, pBu = 2, depth = 16, coeffDepth = 32)
    val tol = 3   // Q1.15 LSBs: allow tiny rounding-order differences
    try {
      simulate(new Engine(cfg)) { c =>
        // defaults
        c.io.start.poke(false.B); c.io.dbgWen.poke(false.B); c.io.cWen.poke(false.B)
        c.io.nCur.poke(4.U); c.io.scale.poke(true.B); c.io.coeffBase.poke(0.U)
        c.io.dbgWidx.poke(0.U); c.io.dbgRidx.poke(0.U)
        c.clock.step()

        // ---- load CRAM: address k*8 + coeff -> 2x2 matrix ----
        for ((entry, a) <- NumGolden.cram.zipWithIndex) {
          val (c00, c01, c10, c11) = entry
          c.io.cWen.poke(true.B); c.io.cWaddr.poke(a.U)
          c.io.cWval(0).re.poke(c00._1.S); c.io.cWval(0).im.poke(c00._2.S)
          c.io.cWval(1).re.poke(c01._1.S); c.io.cWval(1).im.poke(c01._2.S)
          c.io.cWval(2).re.poke(c10._1.S); c.io.cWval(2).im.poke(c10._2.S)
          c.io.cWval(3).re.poke(c11._1.S); c.io.cWval(3).im.poke(c11._2.S)
          c.clock.step()
        }
        c.io.cWen.poke(false.B)

        // ---- load input by logical index ----
        for ((xi, i) <- NumGolden.x.zipWithIndex) {
          c.io.dbgWen.poke(true.B); c.io.dbgWidx.poke(i.U)
          c.io.dbgWval.re.poke(xi._1.S); c.io.dbgWval.im.poke(xi._2.S)
          c.clock.step()
        }
        c.io.dbgWen.poke(false.B)
        c.clock.step()

        // ---- run ----
        c.io.start.poke(true.B); c.clock.step()
        c.io.start.poke(false.B)
        var guard = 0
        while (!c.io.done.peek().litToBoolean && guard < 100) { c.clock.step(); guard += 1 }
        c.clock.step()

        // ---- read back (bit-reversed order) and compare ----
        for ((ye, i) <- NumGolden.y.zipWithIndex) {
          c.io.dbgRidx.poke(i.U)
          c.clock.step()                 // SyncReadMem: data valid next cycle
          val gotRe = c.io.dbgRval.re.peek().litValue.toInt
          val gotIm = c.io.dbgRval.im.peek().litValue.toInt
          assert(math.abs(gotRe - ye._1) <= tol,
            s"re[$i]: got $gotRe expected ${ye._1}")
          assert(math.abs(gotIm - ye._2) <= tol,
            s"im[$i]: got $gotIm expected ${ye._2}")
        }
      }
    } catch {
      case e: Exception if e.getMessage != null &&
          (e.getMessage.contains("Unexpected message") ||
           e.getMessage.contains("verilator not found")) =>
        cancel(s"simulator backend unavailable: ${e.getMessage}")
    }
  }
}
