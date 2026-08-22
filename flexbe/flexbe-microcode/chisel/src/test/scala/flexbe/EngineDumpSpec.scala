// Diagnostic: run the Engine numeric scenario, then dump every BSPM logical
// index and compare to the golden output element-by-element, printing each so we
// can see which indices are wrong and whether it is a datapath or readback fault.
package flexbe

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EngineDumpSpec extends AnyFlatSpec with Matchers {
  "Engine" should "dump BSPM after a 16-point FFT" in {
    val cfg = new EngineConfig(nMax = 4, pBu = 2, depth = 16, coeffDepth = 32)
    try {
      simulate(new Engine(cfg)) { c =>
        c.io.start.poke(false.B); c.io.dbgWen.poke(false.B); c.io.cWen.poke(false.B)
        c.io.nCur.poke(4.U); c.io.scale.poke(true.B); c.io.coeffBase.poke(0.U)
        c.io.dbgWidx.poke(0.U); c.io.dbgRidx.poke(0.U); c.clock.step()

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

        for ((xi, i) <- NumGolden.x.zipWithIndex) {
          c.io.dbgWen.poke(true.B); c.io.dbgWidx.poke(i.U)
          c.io.dbgWval.re.poke(xi._1.S); c.io.dbgWval.im.poke(xi._2.S)
          c.clock.step()
        }
        c.io.dbgWen.poke(false.B); c.clock.step()

        // read back BEFORE running, to confirm load + readback path is clean
        println("--- readback of loaded input (logical index) ---")
        for (i <- 0 until 16) {
          c.io.dbgRidx.poke(i.U); c.clock.step()
          val re = c.io.dbgRval.re.peek().litValue
          val im = c.io.dbgRval.im.peek().litValue
          println(f"  in[$i%2d] got=($re,$im) expected=(${NumGolden.x(i)._1},${NumGolden.x(i)._2})")
        }

        c.io.start.poke(true.B); c.clock.step(); c.io.start.poke(false.B)
        var g = 0; while (!c.io.done.peek().litToBoolean && g < 100) { c.clock.step(); g += 1 }
        c.clock.step()

        println("--- readback after run (bit-reversed order) ---")
        for (i <- 0 until 16) {
          c.io.dbgRidx.poke(i.U); c.clock.step()
          val re = c.io.dbgRval.re.peek().litValue.toInt
          val im = c.io.dbgRval.im.peek().litValue.toInt
          val er = NumGolden.y(i)._1; val ei = NumGolden.y(i)._2
          val mark = if (re == er && im == ei) "" else "  <-- MISMATCH"
          println(f"  y[$i%2d] got=($re,$im) expected=($er,$ei)$mark")
        }
      }
    } catch {
      case e: Exception if e.getMessage != null && e.getMessage.contains("Unexpected message") =>
        cancel("backend unavailable")
    }
  }
}
