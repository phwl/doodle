// Diagnostic: run the FFT one stage at a time and dump BSPM after each, to find
// the FIRST stage whose result diverges from the golden per-stage snapshots.
package flexbe

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class StageDumpSpec extends AnyFlatSpec with Matchers {
  "Engine" should "match golden BSPM after each stage" in {
    val cfg = new EngineConfig(nMax = 4, pBu = 2, depth = 16, coeffDepth = 32)
    try {
      // run once per max-stage, dumping after that stage
      for (maxStage <- 1 until 4) {
        simulate(new Engine(cfg)) { c =>
          c.io.start.poke(false.B); c.io.dbgWen.poke(false.B); c.io.cWen.poke(false.B)
          c.io.nCur.poke(4.U); c.io.scale.poke(true.B); c.io.coeffBase.poke(0.U)
          c.io.dbgMaxStage.poke(maxStage.U)
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

          c.io.start.poke(true.B); c.clock.step(); c.io.start.poke(false.B)
          var g = 0; while (!c.io.done.peek().litToBoolean && g < 200) { c.clock.step(); g += 1 }
          c.clock.step()

          val golden = StageGolden.snaps(maxStage + 1)  // snaps(0)=initial
          var firstBad = -1
          for (i <- 0 until 16) {
            c.io.dbgRidx.poke(i.U); c.clock.step()
            val re = c.io.dbgRval.re.peek().litValue.toInt
            val im = c.io.dbgRval.im.peek().litValue.toInt
            if ((re, im) != golden(i) && firstBad < 0) firstBad = i
          }
          if (firstBad < 0) println(s"stage $maxStage: BSPM matches golden")
          else {
            c.io.dbgRidx.poke(firstBad.U); c.clock.step()
            val re = c.io.dbgRval.re.peek().litValue.toInt
            val im = c.io.dbgRval.im.peek().litValue.toInt
            println(f"stage $maxStage: FIRST MISMATCH at index $firstBad " +
                    f"got=($re,$im) expected=${golden(firstBad)}")
          }
        }
      }
    } catch {
      case e: Exception if e.getMessage != null && e.getMessage.contains("Unexpected message") =>
        cancel("backend unavailable")
    }
  }
}
