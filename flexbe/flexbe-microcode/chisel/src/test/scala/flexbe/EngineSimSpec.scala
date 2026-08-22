// =============================================================================
// EngineSimSpec.scala -- clocked FSM test for the Engine.
//
// This one genuinely needs a Verilog backend (Verilator) via ChiselSim.  It is
// kept separate from FlexBESpec so the backend-free golden checks always run;
// if the svsim/Verilator handshake is unavailable or flaky on a given machine,
// the test cancels itself instead of failing the suite.
//
//   sbt "testOnly flexbe.EngineSimSpec"
// =============================================================================
package flexbe

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EngineSimSpec extends AnyFlatSpec with Matchers {

  "Engine" should "sequence one 16-point transform (start -> busy -> done)" in {
    val cfg = new EngineConfig(nMax = 4, pBu = 2, depth = 16, coeffDepth = 16)
    try {
      simulate(new Engine(cfg)) { c =>
        c.io.nCur.poke(4.U); c.io.scale.poke(true.B); c.io.coeffBase.poke(0.U)
        c.io.start.poke(false.B); c.clock.step()

        c.io.start.poke(true.B); c.clock.step()
        c.io.start.poke(false.B)
        c.io.busy.expect(true.B, "engine should be busy after start")

        var sawDone = false; var cycles = 0
        while (!sawDone && cycles < 64) {
          if (c.io.done.peek().litToBoolean) sawDone = true
          c.clock.step(); cycles += 1
        }
        assert(sawDone, "engine did not raise done within 64 cycles")
      }
    } catch {
      case e: Exception if e.getMessage != null &&
          (e.getMessage.contains("Unexpected message") ||
           e.getMessage.contains("verilator not found")) =>
        cancel(s"simulator backend unavailable on this machine: ${e.getMessage}")
    }
  }
}
